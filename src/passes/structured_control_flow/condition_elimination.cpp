#include "sdfg/passes/structured_control_flow/condition_elimination.h"

#include "sdfg/analysis/users.h"
#include "sdfg/deepcopy/structured_sdfg_deep_copy.h"

namespace sdfg {
namespace passes {

bool ConditionElimination::
    eliminate_condition(structured_control_flow::Sequence& root, structured_control_flow::IfElse& match) {
    auto branch = match.at(0);
    auto& branch_root = branch.first;
    auto& branch_condition = branch.second;
    auto& loop = static_cast<structured_control_flow::StructuredLoop&>(branch_root.at(0).first);

    auto loop_indvar = loop.indvar();
    auto loop_init = loop.init();
    auto loop_condition = loop.condition();

    // indvar shall not be read afterwards directly since loop init overrides it
    analysis::Users& users = this->analysis_manager_.get<analysis::Users>();
    analysis::UsersView loop_users(users, branch_root);
    if (users.reads(loop_indvar->get_name()).size() > loop_users.reads(loop_indvar->get_name()).size()) {
        return false;
    }

    // If loop condition == true => condition == true && condition == false => loop condition == false
    auto loop_iter0_condition = symbolic::subs(loop_condition, loop_indvar, loop_init);
    if (symbolic::eq(loop_iter0_condition, branch_condition)) {
        // Insert placeholder before if-else
        auto new_child = this->builder_.add_sequence_before(root, match);

        // Move children of branch to placeholder
        this->builder_.insert_children(new_child.first, branch_root, 0);

        // Remove now empty if-else
        this->builder_.remove_child(root, match);

        return true;
    }

    return false;
};

ConditionElimination::
    ConditionElimination(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
    : StructuredSDFGVisitor(builder, analysis_manager) {

      };


bool ConditionElimination::accept(structured_control_flow::Sequence& parent, structured_control_flow::IfElse& node) {
    if (node.size() != 1) {
        return false;
    }
    auto branch = node.at(0);
    auto& condition = branch.second;

    auto& root = branch.first;
    if (root.size() != 1) {
        return false;
    }

    if (!dynamic_cast<structured_control_flow::StructuredLoop*>(&root.at(0).first)) {
        return false;
    }

    return this->eliminate_condition(parent, node);
};

} // namespace passes
} // namespace sdfg
