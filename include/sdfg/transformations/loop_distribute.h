#pragma once

#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

class LoopDistribute : public Transformation {
    structured_control_flow::StructuredLoop& loop_;

   public:
    LoopDistribute(structured_control_flow::StructuredLoop& loop);

    virtual std::string name() const override;

    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder,
                                analysis::AnalysisManager& analysis_manager) override;

    virtual void apply(builder::StructuredSDFGBuilder& builder,
                       analysis::AnalysisManager& analysis_manager) override;

    virtual void to_json(nlohmann::json& j) const override;

    static LoopDistribute from_json(builder::StructuredSDFGBuilder& builder,
                                    const nlohmann::json& j);
};

}  // namespace transformations
}  // namespace sdfg
