#pragma once

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/data_parallelism_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/passes/pass.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

class TransitionToTaskletConversion : public visitor::StructuredSDFGVisitor {
   private:
    bool can_be_applied(const symbolic::Symbol& lhs, const symbolic::Expression& rhs) const;

    void apply(structured_control_flow::Block& block, const symbolic::Symbol& lhs,
               const symbolic::Expression& rhs);

   public:
    TransitionToTaskletConversion(builder::StructuredSDFGBuilder& builder,
                                  analysis::AnalysisManager& analysis_manager);

    bool accept(structured_control_flow::Sequence& parent,
                structured_control_flow::Sequence& node) override;
};

typedef VisitorPass<TransitionToTaskletConversion> TransitionToTaskletConversionPass;

}  // namespace passes
}  // namespace sdfg