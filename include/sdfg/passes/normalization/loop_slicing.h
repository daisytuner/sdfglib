#pragma once

#include <algorithm>

#include "sdfg/passes/pass.h"
#include "sdfg/transformations/loop_slicing.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

class LoopSlicing : public visitor::StructuredSDFGVisitor {
   private:
    bool can_be_applied(structured_control_flow::Sequence& parent,
                        structured_control_flow::For& loop);

    void apply(structured_control_flow::Sequence& parent, structured_control_flow::For& loop);

   public:
    LoopSlicing(builder::StructuredSDFGBuilder& builder,
                analysis::AnalysisManager& analysis_manager);

    bool accept(structured_control_flow::Sequence& parent,
                structured_control_flow::For& loop) override;
};

typedef VisitorPass<LoopSlicing> LoopSlicingPass;

}  // namespace passes
}  // namespace sdfg
