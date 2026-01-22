#pragma once
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <string>
#include "sdfg/analysis/users.h"
#include "sdfg/structured_sdfg.h"

#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/passes/rpc/rpc_context.h"
#include "sdfg/passes/rpc/rpc_test_context.h"
#include "sdfg/transformations/transformation.h"
#include "sdfg/transformations/utils.h"

namespace sdfg {
namespace transformations {

/**
 * @deprecated This is only here to make the other code compile. it is not to be used by any other library!
 */
struct TransfertuningRecipe {
    nlohmann::json sdfg;
    nlohmann::json sequence;
    std::string region_id;
    double speedup;
    double distance;
};

class LocalTransferTuningTransform : public sdfg::transformations::Transformation {
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
    const sdfg::passes::rpc::RpcContext& rpc_context_;

    std::vector<TransfertuningRecipe>
    query_recipes(sdfg::StructuredSDFG& sdfg, CURL* curl_handle, struct curl_slist* headers);

public:
    LocalTransferTuningTransform(
        const std::string& target,
        const std::string& category,
        sdfg::StructuredSDFG* sdfg,
        const sdfg::analysis::LoopInfo& loop_info,
        const sdfg::passes::rpc::RpcContext& rpc_context = sdfg::passes::rpc::RpcTestContext::default_context()
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
