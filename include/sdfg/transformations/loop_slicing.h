#pragma once

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/data_parallelism_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/deepcopy/structured_sdfg_deep_copy.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

class LoopSlicing : public Transformation {
    structured_control_flow::Sequence& parent_;
    structured_control_flow::For& loop_;

   public:
    LoopSlicing(structured_control_flow::Sequence& parent, structured_control_flow::For& loop);

    virtual std::string name() override;

    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder,
                                analysis::AnalysisManager& analysis_manager) override;

    virtual void apply(builder::StructuredSDFGBuilder& builder,
                       analysis::AnalysisManager& analysis_manager) override;
};

}  // namespace transformations
}  // namespace sdfg
