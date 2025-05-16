#include "sdfg/passes/normalization/conditional_loop_fissioning.h"

namespace sdfg {
namespace passes {

ConditionalLoopFissioning::ConditionalLoopFissioning(builder::StructuredSDFGBuilder& builder,
                                                     analysis::AnalysisManager& analysis_manager)
    : StructuredSDFGVisitor(builder, analysis_manager) {}

bool ConditionalLoopFissioning::can_be_applied(structured_control_flow::Sequence& parent,
                                               structured_control_flow::For& loop) {
    auto& sdfg = builder_.subject();

    auto indvar = loop.indvar();
    auto& body = loop.root();
    if (body.size() != 1) {
        return false;
    }

    auto& transition = body.at(0).second;
    if (!transition.assignments().empty()) {
        return false;
    }
    auto& node = body.at(0).first;
    if (!dynamic_cast<structured_control_flow::IfElse*>(&node)) {
        return false;
    }
    auto if_else_stmt = static_cast<structured_control_flow::IfElse*>(&node);

    std::unordered_set<std::string> condition_symbols;
    for (size_t i = 0; i < if_else_stmt->size(); i++) {
        auto if_else = if_else_stmt->at(i);
        auto condition = if_else.second;
        for (auto& atom : symbolic::atoms(condition)) {
            auto sym = SymEngine::rcp_static_cast<const SymEngine::Symbol>(atom);
            if (symbolic::is_pointer(sym)) {
                return false;
            }
            if (!dynamic_cast<const types::Scalar*>(&sdfg.type(sym->get_name()))) {
                return false;
            }
            condition_symbols.insert(sym->get_name());
        }
    }

    // Check if the conditions are invariant w.r.t. the loop
    auto& all_users = analysis_manager_.get<analysis::Users>();
    analysis::UsersView users(all_users, loop);
    if (!users.views().empty() || !users.moves().empty()) {
        return false;
    }

    auto writes = users.writes();
    for (auto write : writes) {
        if (condition_symbols.find(write->container()) != condition_symbols.end()) {
            return false;
        }
    }

    return true;
};

void ConditionalLoopFissioning::apply(structured_control_flow::Sequence& parent,
                                      structured_control_flow::For& loop) {
    auto& body = loop.root();
    auto& node = body.at(0).first;
    auto if_else_stmt = dynamic_cast<structured_control_flow::IfElse*>(&node);

    auto& outer_if_else = builder_.add_if_else_before(parent, loop).first;
    for (size_t i = 0; i < if_else_stmt->size(); i++) {
        auto branch = if_else_stmt->at(i);
        auto& branch_root = branch.first;
        auto& condition = branch.second;

        auto& outer_root = builder_.add_case(outer_if_else, condition);
        auto& outer_loop = builder_.add_for(outer_root, loop.indvar(), loop.condition(),
                                            loop.init(), loop.update(), {}, loop.debug_info());

        deepcopy::StructuredSDFGDeepCopy copier(builder_, outer_loop.root(), branch_root);
        copier.copy();
    }

    builder_.remove_child(parent, loop);
};

bool ConditionalLoopFissioning::accept(structured_control_flow::Sequence& parent,
                                       structured_control_flow::For& loop) {
    if (this->can_be_applied(parent, loop)) {
        this->apply(parent, loop);
        return true;
    }
    return false;
};

}  // namespace passes
}  // namespace sdfg
