#include "sdfg/passes/code_motion/block_sorting.h"

#include <climits>
#include <string>
#include <unordered_set>
#include <utility>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/library_nodes/stdlib/alloca.h"
#include "sdfg/data_flow/library_nodes/stdlib/calloc.h"
#include "sdfg/data_flow/library_nodes/stdlib/free.h"
#include "sdfg/data_flow/library_nodes/stdlib/malloc.h"
#include "sdfg/data_flow/library_nodes/stdlib/memset.h"
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

    // Bubble up:
    for (size_t i = 0; i < sequence.size() - 1; i++) {
        // Skip assignments
        auto current_child = sequence.at(i);
        if (!current_child.second.empty()) {
            continue;
        }
        auto next_child = sequence.at(i + 1);
        if (!next_child.second.empty()) {
            continue;
        }

        // Next child must be a block
        auto* next_block = dynamic_cast<structured_control_flow::Block*>(&next_child.first);
        if (!next_block) {
            continue;
        }

        // Check if current block has side-effects
        if (auto current_block = dynamic_cast<structured_control_flow::Block*>(&current_child.first)) {
            bool side_effect = false;
            for (auto* libnode : current_block->dataflow().library_nodes()) {
                if (this->is_libnode_side_effect_white_listed(libnode)) {
                    continue;
                }
                if (libnode->side_effect()) {
                    side_effect = true;
                    break;
                }
            }
            if (side_effect) {
                continue;
            }
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

        // Check if libnode/reference can be bubbled up
        if (!this->can_be_bubbled_up(*next_block)) {
            continue;
        }

        // Compare priority and order
        auto [current_prio, current_order] = this->get_prio_and_order(current_child.first);
        auto [next_prio, next_order] = this->get_prio_and_order(next_child.first);
        if (current_prio > next_prio) {
            continue;
        }
        if (current_prio == next_prio && current_order <= next_order) {
            continue;
        }

        // Swap blocks
        builder_.move_child(sequence, i + 1, sequence, i);
        return true; // Restart after modification
    }

    // Bubble down:
    for (size_t i = sequence.size() - 1; i > 0; i--) {
        // Skip assignments
        auto current_child = sequence.at(i);
        if (!current_child.second.empty()) {
            continue;
        }
        auto next_child = sequence.at(i - 1);
        if (!next_child.second.empty()) {
            continue;
        }

        // Next child must be a block
        auto* next_block = dynamic_cast<structured_control_flow::Block*>(&next_child.first);
        if (!next_block) {
            continue;
        }

        // Check if current block has side-effects
        if (auto current_block = dynamic_cast<structured_control_flow::Block*>(&current_child.first)) {
            bool side_effect = false;
            for (auto* libnode : current_block->dataflow().library_nodes()) {
                if (this->is_libnode_side_effect_white_listed(libnode)) {
                    continue;
                }
                if (libnode->side_effect()) {
                    side_effect = true;
                    break;
                }
            }
            if (side_effect) {
                continue;
            }
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

        // Check if libnode can be bubbled down
        if (!this->can_be_bubbled_down(*next_block)) {
            continue;
        }

        // Compare priority and order
        auto [current_prio, current_order] = this->get_prio_and_order(current_child.first);
        auto [next_prio, next_order] = this->get_prio_and_order(next_child.first);
        if (current_prio > next_prio) {
            continue;
        }
        if (current_prio == next_prio && current_order <= next_order) {
            continue;
        }

        // Swap blocks
        builder_.move_child(sequence, i - 1, sequence, i);
        return true; // Restart after modification
    }

    return false;
}

bool BlockSorting::is_libnode_side_effect_white_listed(data_flow::LibraryNode* libnode) {
    return dynamic_cast<stdlib::AllocaNode*>(libnode) || dynamic_cast<stdlib::CallocNode*>(libnode) ||
           dynamic_cast<stdlib::FreeNode*>(libnode) || dynamic_cast<stdlib::MallocNode*>(libnode) ||
           dynamic_cast<stdlib::MemsetNode*>(libnode);
}

bool BlockSorting::can_be_bubbled_up(structured_control_flow::Block& block) {
    if (this->is_reference_block(block)) {
        return true;
    }

    if (this->is_libnode_block(block)) {
        auto* libnode = *block.dataflow().library_nodes().begin();
        return dynamic_cast<stdlib::AllocaNode*>(libnode) || dynamic_cast<stdlib::CallocNode*>(libnode) ||
               dynamic_cast<stdlib::MallocNode*>(libnode) || dynamic_cast<stdlib::MemsetNode*>(libnode);
    }

    return false;
}

bool BlockSorting::can_be_bubbled_down(structured_control_flow::Block& block) {
    if (!this->is_libnode_block(block)) {
        return false;
    }
    auto* libnode = *block.dataflow().library_nodes().begin();
    return dynamic_cast<stdlib::FreeNode*>(libnode);
}

std::pair<int, std::string> BlockSorting::get_prio_and_order(structured_control_flow::ControlFlowNode& node) {
    if (auto* block = dynamic_cast<structured_control_flow::Block*>(&node)) {
        if (this->is_libnode_block(*block)) {
            auto* libnode = *block->dataflow().library_nodes().begin();
            if (dynamic_cast<stdlib::AllocaNode*>(libnode) || dynamic_cast<stdlib::CallocNode*>(libnode) ||
                dynamic_cast<stdlib::FreeNode*>(libnode) || dynamic_cast<stdlib::MallocNode*>(libnode)) {
                auto& oedge = *block->dataflow().out_edges(*libnode).begin();
                auto& dst = static_cast<data_flow::AccessNode&>(oedge.dst());
                return {300, dst.data()};
            } else if (dynamic_cast<stdlib::MemsetNode*>(libnode)) {
                auto& oedge = *block->dataflow().out_edges(*libnode).begin();
                auto& dst = static_cast<data_flow::AccessNode&>(oedge.dst());
                return {200, dst.data()};
            }
        } else if (this->is_reference_block(*block)) {
            return {100, ""};
        }
    }

    return {INT_MIN, ""};
}

bool BlockSorting::is_reference_block(structured_control_flow::Block& block) {
    auto& dfg = block.dataflow();
    if (dfg.nodes().size() != 2) {
        return false;
    }
    if (dfg.edges().size() != 1) {
        return false;
    }
    auto& edge = *dfg.edges().begin();
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
