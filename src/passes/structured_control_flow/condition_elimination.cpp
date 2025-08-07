#include "sdfg/passes/structured_control_flow/condition_elimination.h"

#include "sdfg/analysis/users.h"
#include "sdfg/deepcopy/structured_sdfg_deep_copy.h"

namespace sdfg {
namespace passes {

bool ConditionElimination::eliminate_condition(
    structured_control_flow::Sequence& root,
    structured_control_flow::IfElse& match,
    structured_control_flow::StructuredLoop& loop,
    const symbolic::Condition& condition
) {
    auto loop_indvar = loop.indvar();
    auto loop_init = loop.init();
    auto loop_condition = loop.condition();

    // indvar shall not be read afterwards directly since loop init overrides it
    analysis::Users& users = this->analysis_manager_.get<analysis::Users>();
    analysis::UsersView loop_users(users, match.at(0).first);
    if (users.reads(loop_indvar->get_name()).size() > loop_users.reads(loop_indvar->get_name()).size()) {
        return false;
    }

    // If loop condition == true => condition == true && condition == false => loop condition == false
    auto loop_iter0_condition = symbolic::subs(loop_condition, loop_indvar, loop_init);
    if (symbolic::eq(loop_iter0_condition, condition)) {
        auto& new_seq = this->builder_.add_sequence_before(root, match).first;
        this->builder_.insert(loop, match.at(0).first, new_seq, match.debug_info());
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
    auto& loop = static_cast<structured_control_flow::StructuredLoop&>(root.at(0).first);

    return this->eliminate_condition(parent, node, loop, condition);
};

} // namespace passes
} // namespace sdfg
