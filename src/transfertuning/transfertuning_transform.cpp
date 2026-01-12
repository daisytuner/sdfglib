#include "sdfg/transfertuning/transfertuning_transform.h"

#include <iostream>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <string>

#include "sdfg/optimization_report/pass_report_consumer.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/transformations/transformation.h"
#include "sdfg/tuning/replayer.h"
#include "sdfg/util/utils_curl.h"

namespace sdfg {
namespace transformations {

const std::string TT_ENDPOINT = "https://europe-west1-daisy-367210.cloudfunctions.net/runner/transfertuning";

std::string get_docc_access_token() {
    static std::string token;
    static std::once_flag flag;
    std::call_once(flag, []() {
        // 1. Check environment variable
        const char* env_token = std::getenv("DOCC_ACCESS_TOKEN");
        if (env_token && *env_token) {
            token = env_token;
            return;
        }
        // 2. Check $HOME/.config/docc/token
        const char* home = std::getenv("HOME");
        if (home && *home) {
            std::filesystem::path config_dir = std::filesystem::path(home) / ".config" / "docc";
            std::string path = (config_dir / "token").string();
            std::ifstream infile(path);
            if (infile) {
                std::ostringstream ss;
                ss << infile.rdbuf();
                token = ss.str();
                // Remove trailing newlines/spaces
                token.erase(token.find_last_not_of(" \n\r\t") + 1);
                return;
            }
        }
        // 3. Not found
        std::cerr << "[ERROR] DOCC access token not found in DOCC_ACCESS_TOKEN or "
                     "$HOME/.config/docc/token"
                  << std::endl;
        token = "";
    });
    return token;
}

TransferTuningTransform::TransferTuningTransform(
    const std::string& target,
    const std::string& category,
    sdfg::StructuredSDFG* sdfg,
    const analysis::LoopInfo& loop_info
)
    : target_(target), category_(category), sdfg_(sdfg), loop_info_(loop_info) {}

std::string TransferTuningTransform::name() const { return "TransferTuningTransform"; }

bool TransferTuningTransform::
    can_be_applied(sdfg::builder::StructuredSDFGBuilder& builder, sdfg::analysis::AnalysisManager& analysis_manager) {
    std::string token = get_docc_access_token();
    if (token.empty()) {
        DEBUG_PRINTLN("[ERROR] No DOCC access token available, aborting transfertuning pass");
#ifndef NDEBUG
        throw std::runtime_error("No DOCC access token available for transfertuning.");
#endif
        return false;
    }
    CURL* curl_handle = curl_easy_init();
    if (!curl_handle) {
        DEBUG_PRINTLN("[ERROR] Could not initialize CURL, aborting transfertuning pass");
        return false;
    }

    std::string auth_header = "Authorization: Bearer " + token;
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    headers = curl_slist_append(headers, auth_header.c_str());
    curl_easy_setopt(curl_handle, CURLOPT_HTTPHEADER, headers);

    this->applied_recipe_ = TransfertuningRecipe();
    this->recipes_ = query_recipes(*sdfg_, curl_handle, headers);
    if (this->recipes_.empty()) {
        if (report_) transformations::Transformation::report_->transform_impossible(this, "No neighbors found");
        return false;
    }

    return true;
}

void TransferTuningTransform::
    apply(sdfg::builder::StructuredSDFGBuilder& builder, sdfg::analysis::AnalysisManager& analysis_manager) {
    this->applied_recipe_ = TransfertuningRecipe();

    sdfg::transformations::Replayer replayer;
    for (auto recipe : this->recipes_) {
        try {
            replayer.replay(builder, analysis_manager, recipe.sequence, true);
        } catch (const std::exception& e) {
            continue;
        }
        this->applied_recipe_ = recipe;
#ifndef NDEBUG
        if (report_) {
            nlohmann::json j;
            this->to_json(j);
            transformations::Transformation::report_->transform_applied(j.dump());
        }
        std::cerr << "[INFO] Applied transfertuning recipe with region ID " << recipe.region_id
                  << " and estimated speedup " << recipe.speedup << std::endl;
#endif
        break;
    }
}

std::vector<TransfertuningRecipe> TransferTuningTransform::
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
    HttpResult res = post_json(curl_handle, TT_ENDPOINT + "/get_recipe", payload_str, headers);
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
    if (!parsed.contains("data") || !parsed["data"].is_array() || parsed["data"].empty()) {
        return {};
    }

    std::vector<TransfertuningRecipe> recipes;
    for (auto entry : parsed) {
        recipes.push_back(TransfertuningRecipe{
            entry["sequence"], entry["region_id"], entry["speedup"], entry["vector_distance"].get<double>()
        });
    }

    return recipes;
}


} // namespace transformations
} // namespace sdfg
