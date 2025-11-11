#include "sdfg/passes/code_motion/block_hoisting.h"
#include <cstddef>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/data_flow/library_nodes/stdlib/alloca.h"
#include "sdfg/data_flow/library_nodes/stdlib/memcpy.h"
#include "sdfg/data_flow/library_nodes/stdlib/memmove.h"
#include "sdfg/data_flow/library_nodes/stdlib/memset.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/element.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

BlockHoisting::BlockHoisting(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
    : visitor::NonStoppingStructuredSDFGVisitor(builder, analysis_manager) {};

bool BlockHoisting::accept(structured_control_flow::Map& map_stmt) {
    auto& scope_analysis = analysis_manager_.get<analysis::ScopeAnalysis>();
    auto& parent = static_cast<structured_control_flow::Sequence&>(*scope_analysis.parent_scope(&map_stmt));

    // Extract first child
    auto& body = map_stmt.root();
    if (body.size() == 0) {
        return false;
    }
    auto first_child = body.at(0);
    if (!first_child.second.assignments().empty()) {
        return false;
    }
    auto& first_node = first_child.first;

    if (auto block = dynamic_cast<structured_control_flow::Block*>(&first_node)) {
        if (this->map_invariant_move(parent, map_stmt, *block)) {
            return true;
        }
        if (this->map_invariant_view(parent, map_stmt, *block)) {
            return true;
        }
        if (this->map_invariant_libnode(parent, map_stmt, *block)) {
            return true;
        }
    }

    return false;
}

bool BlockHoisting::accept(structured_control_flow::For& for_stmt) {
    auto& scope_analysis = analysis_manager_.get<analysis::ScopeAnalysis>();
    auto& parent = static_cast<structured_control_flow::Sequence&>(*scope_analysis.parent_scope(&for_stmt));

    // Extract first child
    auto& body = for_stmt.root();
    if (body.size() == 0) {
        return false;
    }
    auto first_child = body.at(0);
    if (!first_child.second.assignments().empty()) {
        return false;
    }
    auto& first_node = first_child.first;

    if (auto block = dynamic_cast<structured_control_flow::Block*>(&first_node)) {
        if (this->for_invariant_move(parent, for_stmt, *block)) {
            return true;
        }
        if (this->for_invariant_view(parent, for_stmt, *block)) {
            return true;
        }
        if (this->for_invariant_libnode(parent, for_stmt, *block)) {
            return true;
        }
    }

    return false;
}

bool BlockHoisting::accept(structured_control_flow::IfElse& if_else) {
    // Ignore incomplete branches for now
    if (if_else.size() == 0 || !if_else.is_complete()) {
        return false;
    }

    auto& scope_analysis = analysis_manager_.get<analysis::ScopeAnalysis>();
    auto& parent = static_cast<structured_control_flow::Sequence&>(*scope_analysis.parent_scope(&if_else));

    // Extract the first block of first case
    if (if_else.at(0).first.size() == 0) {
        return false;
    }
    auto first_child = if_else.at(0).first.at(0);
    if (!first_child.second.assignments().empty()) {
        return false;
    }
    auto* first_block = dynamic_cast<structured_control_flow::Block*>(&first_child.first);
    if (!first_block) {
        return false;
    }

    // Collect the first block of all other cases
    std::vector<structured_control_flow::Block*> other_blocks;
    for (size_t i = 1; i < if_else.size(); i++) {
        if (if_else.at(i).first.size() == 0) {
            return false;
        }
        auto other_child = if_else.at(i).first.at(0);
        if (!other_child.second.assignments().empty()) {
            return false;
        }
        auto* other_block = dynamic_cast<structured_control_flow::Block*>(&other_child.first);
        if (!other_block) {
            return false;
        }
        other_blocks.push_back(other_block);
    }

    // Check the first block of all cases for invariant move / view
    if (this->is_invariant_move(if_else.at(0).first, first_block->dataflow())) {
        for (size_t i = 0; i < other_blocks.size(); i++) {
            if (!this->is_invariant_move(if_else.at(i + 1).first, other_blocks[i]->dataflow())) {
                return false;
            }
            if (!this->equal_moves(*first_block, *other_blocks[i])) {
                return false;
            }
        }
        this->if_else_extract_invariant(parent, if_else);
        return true;
    } else if (this->is_invariant_view(if_else.at(0).first, first_block->dataflow())) {
        for (size_t i = 0; i < other_blocks.size(); i++) {
            if (!this->is_invariant_view(if_else.at(i + 1).first, other_blocks[i]->dataflow())) {
                return false;
            }
            if (!this->equal_views(*first_block, *other_blocks[i])) {
                return false;
            }
        }
        this->if_else_extract_invariant(parent, if_else);
        return true;
    }

    return false;
}

bool BlockHoisting::is_libnode_allowed(
    structured_control_flow::Sequence& body, data_flow::DataFlowGraph& dfg, data_flow::LibraryNode* libnode
) {
    if (dynamic_cast<stdlib::AllocaNode*>(libnode)) {
        return true;
    } else if (dynamic_cast<stdlib::MemcpyNode*>(libnode)) {
        return true;
    } else if (dynamic_cast<stdlib::MemmoveNode*>(libnode)) {
        return true;
    } else if (dynamic_cast<stdlib::MemsetNode*>(libnode)) {
        return true;
    } else {
        return false;
    }
}

bool BlockHoisting::is_invariant_move(
    structured_control_flow::Sequence& body, data_flow::DataFlowGraph& dfg, bool no_loop_carried_dependencies
) {
    if (dfg.nodes().size() != 2) {
        return false;
    }
    if (dfg.edges().size() != 1) {
        return false;
    }
    auto& edge = *dfg.edges().begin();
    if (edge.type() != data_flow::MemletType::Dereference_Src) {
        return false;
    }

    if (no_loop_carried_dependencies) {
        return true;
    }
    auto& src = static_cast<data_flow::AccessNode&>(edge.src());

    auto& users_analysis = analysis_manager_.get<analysis::Users>();
    analysis::UsersView body_view(users_analysis, body);
    if (!body_view.writes(src.data()).empty() || !body_view.moves(src.data()).empty()) {
        return false;
    }

    return true;
}

bool BlockHoisting::is_invariant_view(
    structured_control_flow::Sequence& body,
    data_flow::DataFlowGraph& dfg,
    symbolic::Symbol indvar,
    bool no_loop_carried_dependencies
) {
    if (dfg.nodes().size() != 2) {
        return false;
    }
    if (dfg.edges().size() != 1) {
        return false;
    }
    auto& edge = *dfg.edges().begin();
    if (edge.type() != data_flow::MemletType::Reference) {
        return false;
    }

    if (!indvar.is_null()) {
        auto& subset = edge.subset();
        for (const auto& dim : subset) {
            if (symbolic::uses(dim, indvar)) {
                return false;
            }
        }
    }

    if (no_loop_carried_dependencies) {
        return true;
    }
    auto& src = static_cast<data_flow::AccessNode&>(edge.src());

    auto& users_analysis = analysis_manager_.get<analysis::Users>();
    analysis::UsersView body_view(users_analysis, body);
    if (!body_view.writes(src.data()).empty() || !body_view.moves(src.data()).empty()) {
        return false;
    }

    return true;
}

bool BlockHoisting::is_invariant_libnode(
    structured_control_flow::Sequence& body,
    data_flow::DataFlowGraph& dfg,
    symbolic::Symbol indvar,
    bool no_loop_carried_dependencies
) {
    if (dfg.library_nodes().size() != 1) {
        return false;
    }
    if (dfg.tasklets().size() != 0) {
        return false;
    }
    auto* libnode = *dfg.library_nodes().begin();
    if (!libnode) {
        return false;
    }
    if (dfg.data_nodes().size() != libnode->outputs().size() + libnode->inputs().size()) {
        return false;
    }

    if (!this->is_libnode_allowed(body, dfg, libnode)) {
        return false;
    }

    if (!indvar.is_null()) {
        if (libnode->symbols().contains(indvar)) {
            return false;
        }
        for (auto& oedge : dfg.out_edges(*libnode)) {
            for (const auto& dim : oedge.subset()) {
                if (symbolic::uses(dim, indvar)) {
                    return false;
                }
            }
        }
        for (auto& iedge : dfg.in_edges(*libnode)) {
            for (const auto& dim : iedge.subset()) {
                if (symbolic::uses(dim, indvar)) {
                    return false;
                }
            }
        }
    }

    if (no_loop_carried_dependencies) {
        return true;
    }

    auto& users_analysis = analysis_manager_.get<analysis::Users>();
    analysis::UsersView body_view(users_analysis, body);
    for (auto& iedge : dfg.in_edges(*libnode)) {
        auto& src = static_cast<data_flow::AccessNode&>(iedge.src());
        if (!body_view.writes(src.data()).empty() || !body_view.moves(src.data()).empty()) {
            return false;
        }
    }

    return true;
}

bool BlockHoisting::equal_moves(structured_control_flow::Block& block1, structured_control_flow::Block& block2) {
    // Edges must have same type and subset
    auto& edge1 = *block1.dataflow().edges().begin();
    auto& edge2 = *block2.dataflow().edges().begin();
    if (edge1.type() != edge2.type()) {
        return false;
    }
    if (edge1.subset().size() != edge2.subset().size()) {
        return false;
    }
    for (size_t i = 0; i < edge1.subset().size(); i++) {
        if (!symbolic::eq(edge1.subset().at(i), edge2.subset().at(i))) {
            return false;
        }
    }

    // Directions must be the same
    if (edge1.src_conn() != edge2.src_conn()) {
        return false;
    }
    if (edge1.dst_conn() != edge2.dst_conn()) {
        return false;
    }

    // src's must have the same containers
    auto& src1 = static_cast<data_flow::AccessNode&>(edge1.src());
    auto& src2 = static_cast<data_flow::AccessNode&>(edge2.src());
    if (src1.data() != src2.data()) {
        return false;
    }

    // dst's must have the same containers
    auto& dst1 = static_cast<data_flow::AccessNode&>(edge1.dst());
    auto& dst2 = static_cast<data_flow::AccessNode&>(edge2.dst());
    if (dst1.data() != dst2.data()) {
        return false;
    }

    return true;
}

bool BlockHoisting::equal_views(structured_control_flow::Block& block1, structured_control_flow::Block& block2) {
    // Edges must have same type and subset
    auto& edge1 = *block1.dataflow().edges().begin();
    auto& edge2 = *block2.dataflow().edges().begin();
    if (edge1.type() != edge2.type()) {
        return false;
    }
    if (edge1.subset().size() != edge2.subset().size()) {
        return false;
    }
    for (size_t i = 0; i < edge1.subset().size(); i++) {
        if (!symbolic::eq(edge1.subset().at(i), edge2.subset().at(i))) {
            return false;
        }
    }

    // src's must have the same containers
    auto& src1 = static_cast<data_flow::AccessNode&>(edge1.src());
    auto& src2 = static_cast<data_flow::AccessNode&>(edge2.src());
    if (src1.data() != src2.data()) {
        return false;
    }

    // dst's must have the same containers
    auto& dst1 = static_cast<data_flow::AccessNode&>(edge1.dst());
    auto& dst2 = static_cast<data_flow::AccessNode&>(edge2.dst());
    if (dst1.data() != dst2.data()) {
        return false;
    }

    return true;
}

bool BlockHoisting::map_invariant_move(
    structured_control_flow::Sequence& parent,
    structured_control_flow::Map& map_stmt,
    structured_control_flow::Block& block
) {
    auto& body = map_stmt.root();
    if (!this->is_invariant_move(body, block.dataflow(), true)) {
        return false;
    }

    size_t map_index = parent.index(map_stmt);
    builder_.move_child(body, 0, parent, map_index);
    return true;
}

bool BlockHoisting::map_invariant_view(
    structured_control_flow::Sequence& parent,
    structured_control_flow::Map& map_stmt,
    structured_control_flow::Block& block
) {
    auto& body = map_stmt.root();
    if (!this->is_invariant_view(body, block.dataflow(), map_stmt.indvar(), true)) {
        return false;
    }

    size_t map_index = parent.index(map_stmt);
    builder_.move_child(body, 0, parent, map_index);
    return true;
}

bool BlockHoisting::map_invariant_libnode(
    structured_control_flow::Sequence& parent,
    structured_control_flow::Map& map_stmt,
    structured_control_flow::Block& block
) {
    // For now, only allow libnode hoisting on sequential maps
    if (map_stmt.schedule_type().value() != ScheduleType_Sequential::value()) {
        return false;
    }

    auto& body = map_stmt.root();
    if (!this->is_invariant_libnode(body, block.dataflow(), map_stmt.indvar(), true)) {
        return false;
    }

    size_t map_index = parent.index(map_stmt);
    builder_.move_child(body, 0, parent, map_index);
    return true;
}

bool BlockHoisting::for_invariant_move(
    structured_control_flow::Sequence& parent,
    structured_control_flow::For& for_stmt,
    structured_control_flow::Block& block
) {
    auto& body = for_stmt.root();
    if (!this->is_invariant_move(body, block.dataflow())) {
        return false;
    }

    size_t for_index = parent.index(for_stmt);
    builder_.move_child(body, 0, parent, for_index);
    return true;
}

bool BlockHoisting::for_invariant_view(
    structured_control_flow::Sequence& parent,
    structured_control_flow::For& for_stmt,
    structured_control_flow::Block& block
) {
    auto& body = for_stmt.root();
    if (!this->is_invariant_view(body, block.dataflow(), for_stmt.indvar())) {
        return false;
    }

    size_t for_index = parent.index(for_stmt);
    builder_.move_child(body, 0, parent, for_index);
    return true;
}

bool BlockHoisting::for_invariant_libnode(
    structured_control_flow::Sequence& parent,
    structured_control_flow::For& for_stmt,
    structured_control_flow::Block& block
) {
    auto& body = for_stmt.root();
    if (!this->is_invariant_libnode(body, block.dataflow(), for_stmt.indvar())) {
        return false;
    }

    size_t map_index = parent.index(for_stmt);
    builder_.move_child(body, 0, parent, map_index);
    return true;
}

void BlockHoisting::
    if_else_extract_invariant(structured_control_flow::Sequence& parent, structured_control_flow::IfElse& if_else) {
    size_t if_else_index = parent.index(if_else);
    builder_.move_child(if_else.at(0).first, 0, parent, if_else_index);

    for (size_t i = 1; i < if_else.size(); i++) {
        builder_.remove_child(if_else.at(i).first, 0);
    }
}

} // namespace passes
} // namespace sdfg
