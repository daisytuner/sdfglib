#include "sdfg/passes/normalization/loop_slicing.h"

namespace sdfg {
namespace passes {

LoopSlicing::LoopSlicing(builder::StructuredSDFGBuilder& builder,
                         analysis::AnalysisManager& analysis_manager)
    : visitor::StructuredSDFGVisitor(builder, analysis_manager) {}

bool LoopSlicing::can_be_applied(structured_control_flow::Sequence& parent,
                                 structured_control_flow::For& loop) {
    transformations::LoopSlicing loop_slicing(parent, loop);
    if (!loop_slicing.can_be_applied(builder_, analysis_manager_)) {
        return false;
    }

    return true;
};

void LoopSlicing::apply(structured_control_flow::Sequence& parent,
                        structured_control_flow::For& loop) {
    transformations::LoopSlicing loop_slicing(parent, loop);
    loop_slicing.apply(builder_, analysis_manager_);
};

bool LoopSlicing::accept(structured_control_flow::Sequence& parent,
                         structured_control_flow::For& loop) {
    if (this->can_be_applied(parent, loop)) {
        this->apply(parent, loop);
        return true;
    }
    return false;
};

}  // namespace passes
}  // namespace sdfg
