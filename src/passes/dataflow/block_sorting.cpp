#include "sdfg/passes/dataflow/block_sorting.h"

#include <unordered_set>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/element.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/types/type.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

BlockSorting::BlockSorting(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
    : visitor::StructuredSDFGVisitor(builder, analysis_manager) {};

bool BlockSorting::accept(structured_control_flow::Sequence& sequence) {
    bool applied = false;
    for (size_t i = 0; i < sequence.size() - 1; i++) {
        auto current_child = sequence.at(i);
        if (!current_child.second.assignments().empty()) {
            continue;
        }
        // Highest-priority: skip if current block is not computational
        if (auto current_block = dynamic_cast<structured_control_flow::Block*>(&current_child.first)) {
            auto& current_dfg = current_block->dataflow();
            if (current_dfg.nodes().size() == 2 && current_dfg.edges().size() == 1) {
                auto& edge = *current_dfg.edges().begin();
                if (edge.type() == data_flow::MemletType::Reference
                    || edge.type() == data_flow::MemletType::Dereference_Src
                    || edge.type() == data_flow::MemletType::Dereference_Dst) {
                        continue;
                }
            }
        }

        auto next_child = sequence.at(i + 1);
        if (!next_child.second.assignments().empty()) {
            continue;
        }
        if (!dynamic_cast<structured_control_flow::Block*>(&current_child.first)) {
            continue;
        }
        auto& next_block = static_cast<structured_control_flow::Block&>(next_child.first);
        
        // Check if next block is a high-priority candidate for sorting
        auto& next_dfg = next_block.dataflow();
        if (next_dfg.nodes().size() != 2) {
            continue;
        }
        if (next_dfg.edges().size() != 1) {
            continue;
        }
        auto& edge = *next_dfg.edges().begin();
        if (edge.type() != data_flow::MemletType::Reference
            && edge.type() != data_flow::MemletType::Dereference_Src
            && edge.type() != data_flow::MemletType::Dereference_Dst) {
            continue;
        }

        // Check if current block has side-effects
        if (auto current_block = dynamic_cast<structured_control_flow::Block*>(&current_child.first)) {
            auto& current_dfg = current_block->dataflow();
            for (auto& libnode : current_dfg.library_nodes()) {
                continue;
            }
        } else {
            continue;
        }

        // Check if happens-before relation allows swapping
        auto& users_analysis = this->analysis_manager_.get<analysis::Users>();
        analysis::UsersView current_users(users_analysis, current_child.first);
        analysis::UsersView next_users(users_analysis, next_child.first);
        bool safe = true;
        for (auto user : next_users.uses()) {
            if (current_users.uses(user->container()).size() > 0) {
                safe = false;
                break;
            }
        }
        if (!safe) {
            continue;
        }

        // Swap blocks
        DEBUG_PRINTLN("BlockSorting: Swapping blocks " << current_child.first.element_id() << " " << next_child.first.element_id());
        builder_.move_child(sequence, i + 1, sequence, i);
        applied = true;
        break; // Restart after modification
    }

    return applied;
}

} // namespace passes
} // namespace sdfg
