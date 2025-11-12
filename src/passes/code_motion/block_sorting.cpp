#include "sdfg/passes/code_motion/block_sorting.h"

#include <unordered_set>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/users.h"
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
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

BlockSorting::BlockSorting(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
    : visitor::StructuredSDFGVisitor(builder, analysis_manager) {};

bool BlockSorting::accept(structured_control_flow::Sequence& sequence) {
    if (sequence.size() == 0) {
        return false;
    }

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
                if (edge.type() == data_flow::MemletType::Reference ||
                    edge.type() == data_flow::MemletType::Dereference_Src ||
                    edge.type() == data_flow::MemletType::Dereference_Dst) {
                    continue;
                }
            }
        }

        auto next_child = sequence.at(i + 1);
        if (!next_child.second.assignments().empty()) {
            continue;
        }
        auto* next_block = dynamic_cast<structured_control_flow::Block*>(&next_child.first);
        if (!next_block) {
            continue;
        }

        // Check if next block is a high-priority candidate for sorting
        if (!this->is_reference_block(*next_block) && !this->is_libnode_block(*next_block)) {
            continue;
        }

        // Check if current block has side-effects
        if (auto current_block = dynamic_cast<structured_control_flow::Block*>(&current_child.first)) {
            auto& current_dfg = current_block->dataflow();
            bool side_effect = false;
            for (auto& libnode : current_dfg.library_nodes()) {
                if (libnode->side_effect()) {
                    side_effect = true;
                    break;
                }
            }
            if (side_effect) {
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
        DEBUG_PRINTLN(
            "BlockSorting: Swapping blocks " << current_child.first.element_id() << " " << next_child.first.element_id()
        );
        builder_.move_child(sequence, i + 1, sequence, i);
        applied = true;
        break; // Restart after modification
    }

    return applied;
}

bool BlockSorting::is_reference_block(structured_control_flow::Block& next_block) {
    auto& next_dfg = next_block.dataflow();
    if (next_dfg.nodes().size() != 2) {
        return false;
    }
    if (next_dfg.edges().size() != 1) {
        return false;
    }
    auto& edge = *next_dfg.edges().begin();
    if (edge.type() != data_flow::MemletType::Reference && edge.type() != data_flow::MemletType::Dereference_Src &&
        edge.type() != data_flow::MemletType::Dereference_Dst) {
        return false;
    }
    return true;
}

bool BlockSorting::is_libnode_block(structured_control_flow::Block& next_block) {
    auto& next_dfg = next_block.dataflow();
    if (next_dfg.library_nodes().size() != 1) {
        return false;
    }
    auto* libnode = *next_dfg.library_nodes().begin();
    if (next_dfg.edges().size() != libnode->inputs().size() + libnode->outputs().size()) {
        return false;
    }
    return true;
}

} // namespace passes
} // namespace sdfg
