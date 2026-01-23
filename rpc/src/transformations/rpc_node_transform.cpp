#include "sdfg/transformations/rpc_node_transform.h"

#include <curl/curl.h>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <string>

#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/optimization_report/pass_report_consumer.h"
#include "sdfg/passes/rpc/rpc_context.h"
#include "sdfg/passes/rpc/rpc_responses.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/transformations/replayer.h"
#include "sdfg/transformations/rpc_node_transform.h"
#include "sdfg/transformations/transformation.h"
#include "sdfg/util/coutouts.h"
#include "sdfg/util/utils_curl.h"

namespace sdfg {
namespace transformations {

std::unique_ptr<passes::rpc::RpcOptResponse>
query_rpc_opt(passes::rpc::RpcOptRequest request, sdfg::passes::rpc::RpcContext& ctx) {
    CURL* curl_handle = curl_easy_init();
    if (!curl_handle) {
        DEBUG_PRINTLN("[ERROR] Could not initialize CURL!");
        return {};
    }

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    // Add all headers provided by the RPC context (auth and optional testing headers).
    auto context_headers = ctx.get_auth_headers();
    for (const auto& [key, value] : context_headers) {
        std::string hdr = key + ": " + value;
        headers = curl_slist_append(headers, hdr.c_str());
    }
    curl_easy_setopt(curl_handle, CURLOPT_HTTPHEADER, headers);

    sdfg::serializer::JSONSerializer serializer;
    nlohmann::json sdfg_json = serializer.serialize(request.sdfg);
    // Construct query payload
    nlohmann::json payload = {{"sdfg", sdfg_json}};
    if (request.category.has_value()) {
        payload["category"] = request.category.value();
    }
    if (request.target.has_value()) {
        payload["target"] = request.target.value();
    }
    if (request.loop_info.has_value()) {
        payload["loop_info"] = analysis::loop_info_to_json(request.loop_info.value());
    }
    std::string payload_str = payload.dump();

    // Send query
    HttpResult res = post_json(curl_handle, ctx.get_remote_address(), payload_str, headers);
    if (res.curl_code != CURLE_OK) {
        DEBUG_PRINTLN("[ERROR] Nearest neighbor query failed " << ": " << res.error_message);
        return {};
    }

    if (res.curl_code != CURLE_OK) {
        DEBUG_PRINTLN("[ERROR] RPC optimization query failed " << res.curl_code << ": " << res.error_message);
        return {};
    }

    std::unique_ptr<passes::rpc::RpcOptResponse> rpc_response;

    try {
        // Parse response
        nlohmann::json parsed;
        try {
            parsed = nlohmann::json::parse(res.body);
        } catch (const std::exception& e) {
            return {};
        }

        rpc_response = std::make_unique<passes::rpc::RpcOptResponse>();

        auto errorJ = parsed.find("error");
        if (errorJ != parsed.end()) {
            DEBUG_PRINTLN("[ERROR] RPC optimization query returned error: " << errorJ->get<std::string>());
            return {};
        }

        auto sdfgResJ = parsed.find("sdfg_result");
        if (sdfgResJ != parsed.end()) {
            auto sdfg_field = sdfgResJ->at("sdfg");
            rpc_response->sdfg_result = {.sdfg = serializer.deserialize(sdfg_field)};
        }

        auto localReplayJ = parsed.find("local_replay");
        if (localReplayJ != parsed.end()) {
            rpc_response->local_replay = {.sequence = localReplayJ->at("sequence")};
        }

        auto metadataJ = parsed.find("metadata");
        if (metadataJ != parsed.end()) {
            auto& meta = rpc_response->metadata;
            auto regionIdJ = metadataJ->find("region_id");
            if (regionIdJ != metadataJ->end()) {
                meta.region_id = regionIdJ->get<std::string>();
            }
            auto speedupJ = metadataJ->find("speedup");
            if (speedupJ != metadataJ->end()) {
                meta.speedup = speedupJ->get<double>();
            }
            auto vectorDistanceJ = metadataJ->find("vector_distance");
            if (vectorDistanceJ != metadataJ->end()) {
                meta.vector_distance = vectorDistanceJ->get<double>();
            }
        }
    } catch (const std::exception& e) {
        DEBUG_PRINTLN("[ERROR] Failed to parse RPC optimization response: " << e.what());
    }

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl_handle);

    return rpc_response;
}

RPCNodeTransform::RPCNodeTransform(
    structured_control_flow::ControlFlowNode& node,
    const std::string& target,
    const std::string& category,
    sdfg::passes::rpc::RpcContext& rpc_context,
    bool dump_steps
)
    : node_(node), target_(target), category_(category), rpc_context_(rpc_context), dump_steps_(dump_steps) {}

std::string RPCNodeTransform::name() const { return "RPCNodeTransform"; }

bool RPCNodeTransform::
    can_be_applied(sdfg::builder::StructuredSDFGBuilder& builder, sdfg::analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = builder.subject();

    // Criterion: Must be outmost loop for now
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    if (!loop_analysis.is_outermost_loop(&this->node_)) {
        return false;
    }

    // Prepare metadata (cutout, loop info)
    // Loop info
    auto loop_info = loop_analysis.loop_info(&this->node_);

    // Cutout SDFG
    std::unique_ptr<sdfg::StructuredSDFG> loop_sdfg = util::cutout(builder, analysis_manager, this->node_);

    this->applied_opt_ = query_rpc_opt(
        {.sdfg = *loop_sdfg,
         .category = this->category_.empty() ? std::nullopt : std::optional<std::string>(this->category_),
         .target = this->target_.empty() ? std::nullopt : std::optional<std::string>(this->target_),
         .loop_info = loop_info},
        rpc_context_
    );
    return this->applied_opt_ != nullptr && (this->applied_opt_->sdfg_result.has_value());
}

void RPCNodeTransform::
    apply(sdfg::builder::StructuredSDFGBuilder& builder, sdfg::analysis::AnalysisManager& analysis_manager) {
    auto& opt = *this->applied_opt_;

    if (!opt.sdfg_result.has_value()) {
        throw std::runtime_error("RPCNodeTransform: No SDFG result to apply.");
    }
    // Apply SDFG result

    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto parent_scope = static_cast<structured_control_flow::Sequence*>(scope_analysis.parent_scope(&this->node_));
    size_t index = parent_scope->index(this->node_);

    // this consumes the SDFG result

    // TODO: add transitions from after loop to tmp_scope
    auto& tmp_scope = builder.add_sequence_before(*parent_scope, this->node_, {}, this->node_.debug_info());
    builder.move_child(*parent_scope, index + 1, tmp_scope);

    builder.move_children(opt.sdfg_result->sdfg->root(), tmp_scope);
    builder.remove_child(*parent_scope, index + 1);

    opt.sdfg_result->sdfg.reset();

    if (opt.local_replay.has_value()) {
        auto recipe = opt.local_replay.value();
        std::cout << "Applied RPC optimization seq to " << this->node_.element_id() << " with speedup "
                  << opt.metadata.speedup << ":\n";
        if (dump_steps_) {
            for (auto& desc : recipe.sequence) {
                bool fail = false;
                auto typeJ = desc.find("transformation_type");
                if (typeJ != desc.end()) {
                    std::cout << "\t" << typeJ->get<std::string>();
                } else {
                    fail = true;
                }
                auto paramsJ = desc.find("parameters");
                if (paramsJ != desc.end()) {
                    std::cout << " (";
                    for (auto& [key, value] : paramsJ->items()) {
                        std::cout << key << "=" << value << ", ";
                    }
                    std::cout << ")";
                }
                if (fail) {
                    std::cout << "\t ## Broken step\n";
                } else {
                    std::cout << "\n";
                }
            }
        }
    } else {
        std::cout << "RPC: Applied plain SDFG to " << this->node_.element_id() << " with speedup "
                  << opt.metadata.speedup << "\n";
    }
}


void RPCNodeTransform::to_json(nlohmann::json& j) const {
    j["transformation_type"] = name();
    nlohmann::json params = {{"target", target_}, {"category", category_}, {"speedup", applied_opt_->metadata.speedup}};
    if (applied_opt_->metadata.region_id.has_value()) {
        params["region_id"] = applied_opt_->metadata.region_id.value();
    }
    if (applied_opt_->metadata.vector_distance.has_value()) {
        params["vector_distance"] = applied_opt_->metadata.vector_distance.value();
    };
    j["parameters"] = params;
}


} // namespace transformations
} // namespace sdfg
