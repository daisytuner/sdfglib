#include "sdfg/transformations/local_transfertuning_transform.h"

#include <iostream>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <string>

#include "sdfg/optimization_report/pass_report_consumer.h"
#include "sdfg/passes/rpc/rpc_responses.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/transformations/replayer.h"
#include "sdfg/transformations/transformation.h"
#include "sdfg/util/utils_curl.h"

namespace sdfg {
namespace transformations {


LocalTransferTuningTransform::LocalTransferTuningTransform(
    const std::string& target,
    const std::string& category,
    sdfg::StructuredSDFG* sdfg,
    const analysis::LoopInfo& loop_info,
    const sdfg::passes::rpc::RpcContext& rpc_context
)
    : target_(target), category_(category), sdfg_(sdfg), loop_info_(loop_info), rpc_context_(rpc_context) {}

std::string LocalTransferTuningTransform::name() const { return "LocalTransferTuningTransform"; }

bool LocalTransferTuningTransform::
    can_be_applied(sdfg::builder::StructuredSDFGBuilder& builder, sdfg::analysis::AnalysisManager& analysis_manager) {
    CURL* curl_handle = curl_easy_init();
    if (!curl_handle) {
        DEBUG_PRINTLN("[ERROR] Could not initialize CURL, aborting transfertuning pass");
        return false;
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

    this->applied_recipe_ = TransfertuningRecipe();
    this->recipes_ = query_recipes(*sdfg_, curl_handle, headers);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl_handle);
    if (this->recipes_.empty()) {
        if (report_) transformations::Transformation::report_->transform_impossible(this, "No neighbors found");
        DEBUG_PRINTLN("[INFO] No transfertuning recipes found for loop");
        return false;
    }

    return true;
}

void LocalTransferTuningTransform::
    apply(sdfg::builder::StructuredSDFGBuilder& builder, sdfg::analysis::AnalysisManager& analysis_manager) {
    this->applied_recipe_ = TransfertuningRecipe();

    sdfg::transformations::Replayer replayer;
    bool success = false;
    for (auto recipe : this->recipes_) {
        try {
            replayer.replay(builder, analysis_manager, recipe.sequence, true);
        } catch (const std::exception& e) {
            continue;
        }
        this->applied_recipe_ = recipe;
        success = true;
        break;
    }
    if (success == false) {
        // If a full SDFG was returned instead of a sequence, rebuild from JSON.
        if (!this->recipes_[0].sdfg.is_null()) {
            sdfg::serializer::JSONSerializer serializer;
            auto final_sdfg = serializer.deserialize(this->recipes_[0].sdfg);
            builder = sdfg::builder::StructuredSDFGBuilder(final_sdfg);
            this->applied_recipe_ = this->recipes_[0];
        } else {
            throw std::runtime_error("Failed to apply any transfertuning recipe even though one was found");
        }
    }
#ifndef NDEBUG
    if (report_) {
        nlohmann::json j;
        this->to_json(j);
        transformations::Transformation::report_->transform_applied(j.dump());
    }
    std::cerr << "[INFO] Applied transfertuning recipe with region ID " << this->applied_recipe_.region_id
              << " and estimated speedup " << this->applied_recipe_.speedup << std::endl;
    for (auto transformation : this->applied_recipe_.sequence) {
        std::cerr << "Applied transformation " << transformation["transformation_type"] << std::endl;
    }
#endif
}

std::vector<TransfertuningRecipe> LocalTransferTuningTransform::
    query_recipes(sdfg::StructuredSDFG& sdfg, CURL* curl_handle, struct curl_slist* headers) {
    sdfg::serializer::JSONSerializer serializer;
    nlohmann::json sdfg_json = serializer.serialize(sdfg);
    // Construct query payload
    nlohmann::json payload = {
        {"sdfg", sdfg_json},
        {"category", this->category_},
        {"target", this->target_},
        {"loop_info", analysis::loop_info_to_json(this->loop_info_)}
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

    std::vector<TransfertuningRecipe> recipes;
    for (auto& entry : parsed["data"]) {
        try {
            nlohmann::json sdfg_field = entry.contains("sdfg") ? entry["sdfg"] : nlohmann::json();
            recipes.push_back(TransfertuningRecipe{
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

    return recipes;
}


} // namespace transformations
} // namespace sdfg
