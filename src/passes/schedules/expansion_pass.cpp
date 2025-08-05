#include "sdfg/passes/schedules/expansion_pass.h"

#include "sdfg/data_flow/library_nodes/math/math_node.h"

namespace sdfg {
namespace passes {

Expansion::Expansion(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
    : visitor::StructuredSDFGVisitor(builder, analysis_manager) {}

bool Expansion::accept(structured_control_flow::Sequence& parent, structured_control_flow::Block& node) {
    auto& dataflow = node.dataflow();
    for (auto& library_node : dataflow.nodes()) {
        if (auto math_node = dynamic_cast<math::MathNode*>(&library_node)) {
            if (math_node->expand(this->builder_, this->analysis_manager_)) {
                return true;
            }
        }
    }
    return false;
}

} // namespace passes
} // namespace sdfg
