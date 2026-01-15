#pragma once
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <string>
#include "sdfg/analysis/users.h"
#include "sdfg/structured_sdfg.h"

#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

struct TransfertuningRecipe {
    nlohmann::json sequence;
    std::string region_id;
    double speedup;
    double distance;
};

class TransferTuningTransform : public sdfg::transformations::Transformation {
private:
    // TT parameters
    std::string target_;
    std::string category_;

    // TT target
    sdfg::StructuredSDFG* sdfg_;
    analysis::LoopInfo loop_info_;

    // TT state
    std::vector<TransfertuningRecipe> recipes_;
    TransfertuningRecipe applied_recipe_;

    std::vector<TransfertuningRecipe>
    query_recipes(sdfg::StructuredSDFG& sdfg, CURL* curl_handle, struct curl_slist* headers);

public:
    TransferTuningTransform(
        const std::string& target,
        const std::string& category,
        sdfg::StructuredSDFG* sdfg,
        const sdfg::analysis::LoopInfo& loop_info
    );

    virtual std::string name() const override;

    virtual bool can_be_applied(
        sdfg::builder::StructuredSDFGBuilder& builder, sdfg::analysis::AnalysisManager& analysis_manager
    ) override;

    virtual void apply(sdfg::builder::StructuredSDFGBuilder& builder, sdfg::analysis::AnalysisManager& analysis_manager)
        override;

    TransfertuningRecipe applied_recipe() const { return applied_recipe_; }

    void to_json(nlohmann::json& j) const override {
        j["transformation_type"] = name();
        j["parameters"] = {
            {"target", target_},
            {"category", category_},
            {"region_id", applied_recipe_.region_id},
            {"speedup", applied_recipe_.speedup},
            {"distance", applied_recipe_.distance}
        };
    }
};


} // namespace transformations
} // namespace sdfg
