#pragma once

#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

class LoopInterchange : public Transformation {
    structured_control_flow::StructuredLoop& outer_loop_;
    structured_control_flow::StructuredLoop& inner_loop_;

public:
    LoopInterchange(
        structured_control_flow::StructuredLoop& outer_loop, structured_control_flow::StructuredLoop& inner_loop
    );

    virtual std::string name() const override;

    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        override;

    virtual void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual void to_json(nlohmann::json& j) const override;

    static LoopInterchange from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);
};

} // namespace transformations
} // namespace sdfg
