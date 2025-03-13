#include "sdfg/passes/normalization/conditional_tasklet_propagation.h"

#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace passes {

ConditionalTaskletPropagation::ConditionalTaskletPropagation(
    builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
    : visitor::StructuredSDFGVisitor(builder, analysis_manager) {}

bool ConditionalTaskletPropagation::can_be_applied(structured_control_flow::Sequence& parent,
                                                   structured_control_flow::IfElse& if_else,
                                                   size_t index) {
    auto& sdfg = builder_.subject();
    auto branch = if_else.at(index);
    auto& root = branch.first;
    auto& condition = branch.second;

    if (root.size() == 0) {
        return false;
    }

    // Criterion: Pure data-flow
    for (size_t i = 0; i < root.size(); i++) {
        auto child = root.at(i);
        if (!dynamic_cast<structured_control_flow::Block*>(&child.first)) {
            return false;
        }
        if (!child.second.assignments().empty()) {
            return false;
        }
    }

    // Criterion: Symbols used in condition are read-only
    auto& all_users = analysis_manager_.get<analysis::Users>();
    analysis::UsersView users(all_users, root);
    if (!users.moves().empty() || !users.views().empty()) {
        return false;
    }

    std::unordered_set<std::string> unsafe_symbols;
    for (auto& atom : symbolic::atoms(condition)) {
        auto sym = SymEngine::rcp_static_cast<const SymEngine::Symbol>(atom);
        if (symbolic::is_pointer(sym)) {
            return false;
        }
        if (!symbolic::is_nvptx(sym)) {
            if (!dynamic_cast<const types::Scalar*>(&sdfg.type(sym->get_name()))) {
                return false;
            }
        }
        if (users.writes(sym->get_name()).size() > 0) {
            return false;
        }
        unsafe_symbols.insert(sym->get_name());
    }

    // Unsafe memory accesses: e.g., if (n > 0) A[n - 1]
    for (auto write : users.writes()) {
        auto& container = write->container();
        auto& type = sdfg.type(container);
        if (!dynamic_cast<const types::Scalar*>(&type) ||
            !types::is_integer(type.primitive_type())) {
            continue;
        }
        unsafe_symbols.insert(container);
    }

    for (auto read : users.reads()) {
        auto& container = read->container();
        if (symbolic::is_nvptx(symbolic::symbol(container))) {
            continue;
        }
        auto& type = sdfg.type(container);
        if (!dynamic_cast<const types::Scalar*>(&type) ||
            !types::is_integer(type.primitive_type())) {
            continue;
        }
        if (!dynamic_cast<data_flow::Memlet*>(read->element())) {
            continue;
        }
        if (unsafe_symbols.find(container) != unsafe_symbols.end()) {
            return false;
        }
    }

    return true;
};

void ConditionalTaskletPropagation::apply(structured_control_flow::Sequence& parent,
                                          structured_control_flow::IfElse& if_else, size_t index,
                                          const symbolic::Condition& condition) {
    auto& sdfg = builder_.subject();

    auto branch = if_else.at(index);
    auto& root = branch.first;

    auto& users = analysis_manager_.get<analysis::Users>();
    analysis::UsersView users_branch(users, root);

    // Determine all writes to externals
    auto locals_vars = users.locals(sdfg, root);
    std::unordered_set<std::string> candidates;
    for (auto write : users_branch.writes()) {
        auto& container = write->container();
        if (locals_vars.find(container) != locals_vars.end()) {
            continue;
        }

        candidates.insert(container);
    }

    // Make writes conditional
    for (auto& write : candidates) {
        auto uses = users_branch.writes(write);
        for (auto& use : uses) {
            auto access_node = static_cast<data_flow::AccessNode*>(use->element());
            auto& graph = access_node->get_parent();
            auto& edge = *graph.in_edges(*access_node).begin();
            auto& tasklet = static_cast<data_flow::Tasklet&>(edge.src());
            tasklet.condition() = symbolic::And(tasklet.condition(), condition);
        }
    }

    // Remove branch
    auto& new_sequence = builder_.add_sequence_before(parent, if_else).first;
    deepcopy::StructuredSDFGDeepCopy copier(builder_, new_sequence, root);
    copier.copy();

    builder_.remove_case(if_else, index);
};

bool ConditionalTaskletPropagation::accept(structured_control_flow::Sequence& parent,
                                           structured_control_flow::IfElse& if_else) {
    for (size_t i = 0; i < if_else.size(); i++) {
        if (!this->can_be_applied(parent, if_else, i)) {
            return false;
        }
    }

    auto condition = symbolic::__true__();
    for (size_t j = 0; j < if_else.size();) {
        auto branch_condition = if_else.at(j).second;
        auto apply_condition = symbolic::And(condition, if_else.at(j).second);
        this->apply(parent, if_else, j, apply_condition);
        condition = symbolic::And(condition, symbolic::Not(branch_condition));
    }

    return true;
};

}  // namespace passes
}  // namespace sdfg
