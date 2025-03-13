#pragma once

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/data_parallelism_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/deepcopy/structured_sdfg_deep_copy.h"
#include "sdfg/passes/pass.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

class ConditionalTaskletPropagation : public visitor::StructuredSDFGVisitor {
   private:
    bool can_be_applied(structured_control_flow::Sequence& parent,
                        structured_control_flow::IfElse& if_else, size_t index);

    void apply(structured_control_flow::Sequence& parent, structured_control_flow::IfElse& if_else,
               size_t index, const symbolic::Condition& condition);

   public:
    ConditionalTaskletPropagation(builder::StructuredSDFGBuilder& builder,
                                  analysis::AnalysisManager& analysis_manager);

    bool accept(structured_control_flow::Sequence& parent,
                structured_control_flow::IfElse& if_else) override;
};

typedef VisitorPass<ConditionalTaskletPropagation> ConditionalTaskletPropagationPass;

}  // namespace passes
}  // namespace sdfg
