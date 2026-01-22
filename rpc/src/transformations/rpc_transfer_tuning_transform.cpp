#include "sdfg/transformations/rpc_transfer_tuning_transform.h"

#include <iostream>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <string>

#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/optimization_report/pass_report_consumer.h"
#include "sdfg/passes/rpc/rpc_responses.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/transformations/replayer.h"
#include "sdfg/transformations/transformation.h"
#include "sdfg/util/coutouts.h"
#include "sdfg/util/utils_curl.h"

namespace sdfg {
namespace transformations {


RPCTransferTuningTransform::RPCTransferTuningTransform(
    structured_control_flow::StructuredLoop& loop,
    const std::string& target,
    const std::string& category,
    const sdfg::passes::rpc::RpcContext& rpc_context
)
    : loop_(loop), target_(target), category_(category), rpc_context_(rpc_context) {}

std::string RPCTransferTuningTransform::name() const { return "RPCTransferTuningTransform"; }

bool RPCTransferTuningTransform::
    can_be_applied(sdfg::builder::StructuredSDFGBuilder& builder, sdfg::analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = builder.subject();

    // Criterion: Must be outmost loop for now
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    if (!loop_analysis.is_outermost_loop(&this->loop_)) {
        return false;
    }

    // Prepare metadata (cutout, loop info)
    // Loop info
    auto loop_info = loop_analysis.loop_info(&this->loop_);

    // Cutout SDFG
    std::unique_ptr<sdfg::StructuredSDFG> loop_sdfg = util::cutout(builder, analysis_manager, this->loop_);

    this->recipes_ = query_recipes(*loop_sdfg, loop_info);
    return this->recipes_.size() > 0;
}

void RPCTransferTuningTransform::
    apply(sdfg::builder::StructuredSDFGBuilder& builder, sdfg::analysis::AnalysisManager& analysis_manager) {
    this->applied_recipe_ = this->recipes_[0];

    sdfg::serializer::JSONSerializer serializer;
    auto recipe_sdfg = serializer.deserialize(this->applied_recipe_.sdfg);

    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto parent_scope = static_cast<structured_control_flow::Sequence*>(scope_analysis.parent_scope(&this->loop_));
    size_t index = parent_scope->index(this->loop_);

    // TODO: add transitions from after loop to tmp_scope
    auto& tmp_scope = builder.add_sequence_before(*parent_scope, this->loop_, {}, this->loop_.debug_info());
    builder.move_child(*parent_scope, index + 1, tmp_scope);

    builder.move_children(recipe_sdfg->root(), tmp_scope);
    builder.remove_child(*parent_scope, index + 1);
    std::cout << "Applied recipe with speedup " << this->applied_recipe_.speedup << "\n";
}

std::vector<TransferTuningRecipe> RPCTransferTuningTransform::
    query_recipes(sdfg::StructuredSDFG& sdfg, analysis::LoopInfo& loop_info) {
    CURL* curl_handle = curl_easy_init();
    if (!curl_handle) {
        DEBUG_PRINTLN("[ERROR] Could not initialize CURL, aborting transfertuning pass");
        return {};
    }

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    // Add all headers provided by the RPC context (auth and optional testing headers).
    auto context_headers = rpc_context_.get_auth_headers();
    for (const auto& [key, value] : context_headers) {
        std::string hdr = key + ": " + value;
        headers = curl_slist_append(headers, hdr.c_str());
    }
    curl_easy_setopt(curl_handle, CURLOPT_HTTPHEADER, headers);

    sdfg::serializer::JSONSerializer serializer;
    nlohmann::json sdfg_json = serializer.serialize(sdfg);
    // Construct query payload
    nlohmann::json payload = {
        {"sdfg", sdfg_json},
        {"category", this->category_},
        {"target", this->target_},
        {"loop_info", analysis::loop_info_to_json(loop_info)}
    };
    std::string payload_str = payload.dump();

    // Send query
    HttpResult res = post_json(curl_handle, this->rpc_context_.get_remote_address(), payload_str, headers);
    if (res.curl_code != CURLE_OK) {
        DEBUG_PRINTLN("[ERROR] Nearest neighbor query failed " << ": " << res.error_message);
        return {};
    }

    // Parse response
    nlohmann::json parsed;
    try {
        parsed = json::parse(res.body);
    } catch (const std::exception& e) {
        return {};
    }

    if (!parsed.contains("data") || parsed["data"].empty()) {
        return {};
    }

    std::vector<TransferTuningRecipe> recipes;
    for (auto& entry : parsed["data"]) {
        try {
            nlohmann::json sdfg_field = entry.contains("sdfg") ? entry["sdfg"] : nlohmann::json();
            recipes.push_back(TransferTuningRecipe{
                sdfg_field,
                entry["sequence"],
                entry["region_id"],
                entry["speedup"],
                entry["vector_distance"].get<double>()
            });
        } catch (...) {
            continue;
        }
    }

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl_handle);
    if (this->recipes_.empty()) {
        if (report_) transformations::Transformation::report_->transform_impossible(this, "No neighbors found");
        DEBUG_PRINTLN("[INFO] No transfertuning recipes found for loop");
        return {};
    }

    return recipes;
}


} // namespace transformations
} // namespace sdfg
