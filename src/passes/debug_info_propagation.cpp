#include "sdfg/passes/debug_info_propagation.h"
#include "sdfg/debug_info.h"

namespace sdfg {
namespace passes {

DebugInfoPropagation::DebugInfoPropagation() : Pass() {}

void DebugInfoPropagation::
    propagate(builder::StructuredSDFGBuilder& builder, structured_control_flow::ControlFlowNode* current) {
    auto current_debug_info = current->debug_info();

    if (auto block = dynamic_cast<structured_control_flow::Block*>(current)) {
        auto& graph = block->dataflow();
        for (auto& node : graph.nodes()) {
            current_debug_info = DebugInfoRegion::
                merge(current_debug_info, node.debug_info(), builder.subject().debug_info().instructions());
        }
        for (auto& edge : graph.edges()) {
            current_debug_info = DebugInfoRegion::
                merge(current_debug_info, edge.debug_info(), builder.subject().debug_info().instructions());
        }
    } else if (auto sequence_stmt = dynamic_cast<structured_control_flow::Sequence*>(current)) {
        for (size_t i = 0; i < sequence_stmt->size(); i++) {
            this->propagate(builder, &sequence_stmt->at(i).first);
            current_debug_info = DebugInfoRegion::merge(
                current_debug_info,
                sequence_stmt->at(i).first.debug_info(),
                builder.subject().debug_info().instructions()
            );
            current_debug_info = DebugInfoRegion::merge(
                current_debug_info,
                sequence_stmt->at(i).second.debug_info(),
                builder.subject().debug_info().instructions()
            );
        }
    } else if (auto if_else_stmt = dynamic_cast<structured_control_flow::IfElse*>(current)) {
        for (size_t i = 0; i < if_else_stmt->size(); i++) {
            this->propagate(builder, &if_else_stmt->at(i).first);
            current_debug_info = DebugInfoRegion::merge(
                current_debug_info,
                if_else_stmt->at(i).first.debug_info(),
                builder.subject().debug_info().instructions()
            );
        }
    } else if (auto while_stmt = dynamic_cast<structured_control_flow::While*>(current)) {
        this->propagate(builder, &while_stmt->root());
        current_debug_info = DebugInfoRegion::
            merge(current_debug_info, while_stmt->root().debug_info(), builder.subject().debug_info().instructions());
    } else if (auto loop_stmt = dynamic_cast<structured_control_flow::StructuredLoop*>(current)) {
        this->propagate(builder, &loop_stmt->root());
        current_debug_info = DebugInfoRegion::
            merge(current_debug_info, loop_stmt->root().debug_info(), builder.subject().debug_info().instructions());
    } else if (auto break_stmt = dynamic_cast<structured_control_flow::Break*>(current)) {
        current_debug_info = DebugInfoRegion::
            merge(current_debug_info, break_stmt->debug_info(), builder.subject().debug_info().instructions());
    } else if (auto continue_stmt = dynamic_cast<structured_control_flow::Continue*>(current)) {
        current_debug_info = DebugInfoRegion::
            merge(current_debug_info, continue_stmt->debug_info(), builder.subject().debug_info().instructions());
    } else if (auto return_stmt = dynamic_cast<structured_control_flow::Return*>(current)) {
        current_debug_info = DebugInfoRegion::
            merge(current_debug_info, return_stmt->debug_info(), builder.subject().debug_info().instructions());
    } else {
        throw InvalidSDFGException("Unsupported control flow node type");
    }

    current->set_debug_info(current_debug_info);
}

bool DebugInfoPropagation::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    this->propagate(builder, &builder.subject().root());
    return true;
}

std::string DebugInfoPropagation::name() { return "DebugInfoPropagation"; }

} // namespace passes
} // namespace sdfg
