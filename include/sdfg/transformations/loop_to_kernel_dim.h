#pragma once

#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

class LoopToKernelDim : public Transformation {
    structured_control_flow::Sequence& parent_;
    structured_control_flow::StructuredLoop& loop_;

   public:
    LoopToKernelDim(structured_control_flow::Sequence& parent,
                    structured_control_flow::StructuredLoop& loop);

    virtual std::string name() override;

    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder,
                                analysis::AnalysisManager& analysis_manager) override;

    virtual void apply(builder::StructuredSDFGBuilder& builder,
                       analysis::AnalysisManager& analysis_manager) override;
};

}  // namespace transformations
}  // namespace sdfg
