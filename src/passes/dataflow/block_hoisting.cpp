#include "sdfg/passes/dataflow/block_hoisting.h"

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
    }

    return false;
}

bool BlockHoisting::map_invariant_move(
    structured_control_flow::Sequence& parent,
    structured_control_flow::Map& map_stmt,
    structured_control_flow::Block& block
) {
    size_t map_index = parent.index(map_stmt);
    auto& body = map_stmt.root();
    auto& dfg = block.dataflow();

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

    builder_.move_child(body, 0, parent, map_index);
    return true;
}

bool BlockHoisting::map_invariant_view(
    structured_control_flow::Sequence& parent,
    structured_control_flow::Map& map_stmt,
    structured_control_flow::Block& block
) {
    size_t map_index = parent.index(map_stmt);
    auto& body = map_stmt.root();
    auto& dfg = block.dataflow();

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
    auto& subset = edge.subset();
    for (const auto& dim : subset) {
        if (symbolic::uses(dim, map_stmt.indvar())) {
            return false;
        }
    }

    builder_.move_child(body, 0, parent, map_index);
    return true;
}

bool BlockHoisting::for_invariant_move(
    structured_control_flow::Sequence& parent,
    structured_control_flow::For& for_stmt,
    structured_control_flow::Block& block
) {
    size_t for_index = parent.index(for_stmt);
    auto& body = for_stmt.root();
    auto& dfg = block.dataflow();

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
    auto& src = static_cast<data_flow::AccessNode&>(edge.src());

    auto& users_analysis = analysis_manager_.get<analysis::Users>();
    analysis::UsersView body_view(users_analysis, body);
    if (!body_view.writes(src.data()).empty()
        || !body_view.moves(src.data()).empty()) {
        return false;
    }

    builder_.move_child(body, 0, parent, for_index);
    return true;
}

bool BlockHoisting::for_invariant_view(
    structured_control_flow::Sequence& parent,
    structured_control_flow::For& for_stmt,
    structured_control_flow::Block& block
) {
    size_t for_index = parent.index(for_stmt);
    auto& body = for_stmt.root();
    auto& dfg = block.dataflow();

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
    auto& src = static_cast<data_flow::AccessNode&>(edge.src());

    auto& users_analysis = analysis_manager_.get<analysis::Users>();
    analysis::UsersView body_view(users_analysis, body);
    if (!body_view.writes(src.data()).empty()
        || !body_view.moves(src.data()).empty()) {
        return false;
    }

    auto& subset = edge.subset();
    for (const auto& dim : subset) {
        if (symbolic::uses(dim, for_stmt.indvar())) {
            return false;
        }
    }

    builder_.move_child(body, 0, parent, for_index);
    return true;
}

} // namespace passes
} // namespace sdfg
