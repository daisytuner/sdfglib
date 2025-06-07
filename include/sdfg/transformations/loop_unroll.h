#pragma once

#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

class LoopUnroll : public Transformation {
    structured_control_flow::Sequence& parent_;
    structured_control_flow::StructuredLoop& loop_;

   public:
    LoopUnroll(structured_control_flow::Sequence& parent,
               structured_control_flow::StructuredLoop& loop);

    virtual std::string name() override;

    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder,
                                analysis::AnalysisManager& analysis_manager) override;

    virtual void apply(builder::StructuredSDFGBuilder& builder,
                       analysis::AnalysisManager& analysis_manager) override;
};

}  // namespace transformations
}  // namespace sdfg