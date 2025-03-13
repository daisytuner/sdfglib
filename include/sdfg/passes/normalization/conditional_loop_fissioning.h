#pragma once

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/data_parallelism_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/deepcopy/structured_sdfg_deep_copy.h"
#include "sdfg/passes/pass.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

class ConditionalLoopFissioning : public visitor::StructuredSDFGVisitor {
   private:
    bool can_be_applied(structured_control_flow::Sequence& parent,
                        structured_control_flow::For& loop);

    void apply(structured_control_flow::Sequence& parent, structured_control_flow::For& loop);

   public:
    ConditionalLoopFissioning(builder::StructuredSDFGBuilder& builder,
                              analysis::AnalysisManager& analysis_manager);

    bool accept(structured_control_flow::Sequence& parent,
                structured_control_flow::For& loop) override;
};

typedef VisitorPass<ConditionalLoopFissioning> ConditionalLoopFissioningPass;

}  // namespace passes
}  // namespace sdfg
