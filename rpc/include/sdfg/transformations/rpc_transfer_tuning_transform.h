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
struct TransferTuningRecipe {
    nlohmann::json sdfg;
    nlohmann::json sequence;
    std::string region_id;
    double speedup;
    double distance;
};

class RPCTransferTuningTransform : public sdfg::transformations::Transformation {
private:
    structured_control_flow::StructuredLoop& loop_;
    std::string target_;
    std::string category_;

    const sdfg::passes::rpc::RpcContext& rpc_context_;

    // TT state
    std::vector<TransferTuningRecipe> recipes_;
    TransferTuningRecipe applied_recipe_;

    std::vector<TransferTuningRecipe> query_recipes(sdfg::StructuredSDFG& sdfg, analysis::LoopInfo& loop_info);

public:
    RPCTransferTuningTransform(
        structured_control_flow::StructuredLoop& loop,
        const std::string& target,
        const std::string& category,
        const sdfg::passes::rpc::RpcContext& rpc_context = sdfg::passes::rpc::RpcTestContext::default_context()
    );

    virtual std::string name() const override;

    virtual bool can_be_applied(
        sdfg::builder::StructuredSDFGBuilder& builder, sdfg::analysis::AnalysisManager& analysis_manager
    ) override;

    virtual void apply(sdfg::builder::StructuredSDFGBuilder& builder, sdfg::analysis::AnalysisManager& analysis_manager)
        override;

    TransferTuningRecipe applied_recipe() const { return applied_recipe_; }

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
