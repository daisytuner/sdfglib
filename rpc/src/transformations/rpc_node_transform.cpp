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

std::variant<std::unique_ptr<passes::rpc::RpcOptResponse>, std::string>
query_rpc_opt(passes::rpc::RpcOptRequest request, sdfg::passes::rpc::RpcContext& ctx) {
    CURL* curl_handle = curl_easy_init();
    if (!curl_handle) {
        std::cerr << "[ERROR] Could not initialize CURL!" << std::endl;
        return {"CurlInit"};
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
        std::cerr << "[ERROR] RPC optimization query failed " << res.curl_code << ": " << res.error_message
                  << std::endl;
        return {"CurlReq"};
    }

    if (res.http_status == 401) {
        nlohmann::json parsed;
        try {
            parsed = nlohmann::json::parse(res.body);
            auto message = parsed.at("message").get<std::string>();
            std::cerr << "[ERROR] RPC optimization query authentication issue: " << message << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] RPC optimization query failed with " << res.http_status << ":" << res.body
                      << std::endl;
            return {"HttpAuth"};
        }
    } else if (res.http_status > 299 || res.http_status < 200) {
        std::cerr << "[ERROR] RPC optimization query failed with " << res.http_status << ":" << res.body << std::endl;
        return {"HttpReq"};
    }

    std::unique_ptr<passes::rpc::RpcOptResponse> rpc_response;

    try {
        // Parse response
        nlohmann::json parsed;
        try {
            parsed = nlohmann::json::parse(res.body);
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] RPC optimization response failed to parse: " << e.what() << std::endl;
            return {"InvalidJsonResp"};
        }

        rpc_response = std::make_unique<passes::rpc::RpcOptResponse>();

        auto errorJ = parsed.find("error");
        if (errorJ != parsed.end()) {
            std::cerr << "[ERROR] RPC optimization query returned error: " << errorJ->get<std::string>() << std::endl;
            rpc_response->error = errorJ->get<std::string>();
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
        std::cerr << "[ERROR] Failed to parse RPC optimization response: " << e.what() << std::endl;
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

bool RPCNodeTransform::can_apply_opt_sdfg(std::optional<passes::rpc::RpcSdfgResult>& opt_sdfg) const {
    return opt_sdfg.has_value();
}

bool RPCNodeTransform::can_apply_replay(std::optional<passes::rpc::RpcLocalReplayRecipe>& replay) const {
    if (replay.has_value()) {
        return false;
    } else {
        return false;
    }
}

std::string RPCNodeTransform::get_node_id_str() const { return std::to_string(this->node_.element_id()); }

bool RPCNodeTransform::
    can_be_applied(sdfg::builder::StructuredSDFGBuilder& builder, sdfg::analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = builder.subject();

    // Criterion: Must be outmost loop for now
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    if (!loop_analysis.is_outermost_loop(&this->node_)) {
        if (report_) {
            report_->transform_impossible(this->name(), "Not outermost loop (" + get_node_id_str() + ")");
        }
        return false;
    }

    // Prepare metadata (cutout, loop info)
    // Loop info
    auto loop_info = loop_analysis.loop_info(&this->node_);

    // Cutout SDFG
    std::unique_ptr<sdfg::StructuredSDFG> loop_sdfg = util::cutout(builder, analysis_manager, this->node_);

    auto opt_resp = query_rpc_opt(
        {.sdfg = *loop_sdfg,
         .category = this->category_.empty() ? std::nullopt : std::optional<std::string>(this->category_),
         .target = this->target_.empty() ? std::nullopt : std::optional<std::string>(this->target_),
         .loop_info = loop_info},
        rpc_context_
    );
    if (std::holds_alternative<std::unique_ptr<passes::rpc::RpcOptResponse>>(opt_resp)) {
        this->opt_resp_ = std::move(std::get<std::unique_ptr<passes::rpc::RpcOptResponse>>(opt_resp));
    }
    bool can_apply = this->opt_resp_ != nullptr && (can_apply_opt_sdfg(this->opt_resp_->sdfg_result) ||
                                                    can_apply_replay(this->opt_resp_->local_replay));
    if (report_) {
        if (!can_apply) {
            std::string error_msg;
            if (std::holds_alternative<std::string>(opt_resp)) {
                error_msg = std::get<std::string>(opt_resp);
            } else if (this->opt_resp_->error.has_value()) {
                error_msg = this->opt_resp_->error.value();
            }
            report_->transform_impossible(
                this->name(), "No opt. SDFG received (" + get_node_id_str() + ", " + error_msg + ")"
            );
        }
    }
    return can_apply;
}

void RPCNodeTransform::
    apply(sdfg::builder::StructuredSDFGBuilder& builder, sdfg::analysis::AnalysisManager& analysis_manager) {
    auto& opt = *this->opt_resp_;

    if (!opt.sdfg_result.has_value() && !opt.local_replay.has_value()) {
        throw std::runtime_error("RPCNodeTransform: No SDFG result or replay to apply.");
    }

    int element_id = this->node_.element_id();

    if (opt.sdfg_result.has_value()) {
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
    } else if (opt.local_replay.has_value()) {
        try {
            Replayer replayer;
            replayer.replay(builder, analysis_manager, opt.local_replay.value().sequence, false);
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] Failed to replay rpc optimization: " << e.what() << std::endl;
            if (report_) {
                report_
                    ->transform_impossible(this->name(), "Failed to replay local optimization: " + std::string(e.what()));
            }
            return;
        }
    }

    if (report_) {
        nlohmann::json j;
        this->to_json(j);
        report_->transform_applied(this->name(), j);
    }

    if (opt.local_replay.has_value()) {
        auto recipe = opt.local_replay.value();
        std::cout << "Applied RPC optimization seq to " << element_id << " with speedup " << opt.metadata.speedup
                  << ":\n";
        if (dump_steps_) {
            if (recipe.sequence.empty()) {
                std::cerr << "Server sent empty sequence!" << std::endl;
            } else {
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
        }
    } else {
        std::cout << "RPC: Applied plain SDFG to " << this->node_.element_id() << " with speedup "
                  << opt.metadata.speedup << "\n";
    }
}


void RPCNodeTransform::to_json(nlohmann::json& j) const {
    j["transformation_type"] = name();
    nlohmann::json params = {{"target", target_}, {"category", category_}, {"speedup", opt_resp_->metadata.speedup}};
    if (opt_resp_->metadata.region_id.has_value()) {
        params["region_id"] = opt_resp_->metadata.region_id.value();
    }
    if (opt_resp_->metadata.vector_distance.has_value()) {
        params["vector_distance"] = opt_resp_->metadata.vector_distance.value();
    };
    j["parameters"] = params;
}


} // namespace transformations
} // namespace sdfg
