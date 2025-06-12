#include "sdfg/passes/symbolic/symbol_propagation.h"

#include "sdfg/analysis/happens_before_analysis.h"

namespace sdfg {
namespace passes {

SymbolPropagation::SymbolPropagation()
    : Pass() {

      };

std::string SymbolPropagation::name() { return "SymbolPropagation"; };

bool SymbolPropagation::run_pass(builder::StructuredSDFGBuilder& builder,
                                 analysis::AnalysisManager& analysis_manager) {
    bool applied = false;

    auto& sdfg = builder.subject();
    auto& users = analysis_manager.get<analysis::Users>();
    auto& happens_before = analysis_manager.get<analysis::HappensBeforeAnalysis>();
    for (auto& name : sdfg.containers()) {
        // Criterion: Only transients
        if (!sdfg.is_transient(name)) {
            continue;
        }

        // Criterion: Only integers
        auto& type = builder.subject().type(name);
        auto scalar = dynamic_cast<const types::Scalar*>(&type);
        if (!scalar || !types::is_integer(scalar->primitive_type())) {
            continue;
        }

        // The symbol will become the LHS (to be replaced)
        auto lhs = symbolic::symbol(name);

        // Collect all reads of the symbol w.r.t to their writes
        auto raw_groups = happens_before.reads_after_write_groups(name);
        for (auto& entry : raw_groups) {
            // If not exclusive write, skip
            if (entry.second.size() != 1) {
                continue;
            }

            // Criterion: Cannot propagate symbolic expression into an access node
            auto read = entry.first;
            if (dynamic_cast<data_flow::AccessNode*>(read->element())) {
                continue;
            }

            // Criterion: Write must be a transition
            auto write = *entry.second.begin();
            auto transition = dynamic_cast<structured_control_flow::Transition*>(write->element());
            if (!transition) {
                continue;
            }

            // We now define the rhs (to be propagated expression)
            auto rhs = transition->assignments().at(lhs);

            // Criterion: RHS is not trivial and not recursive
            if (symbolic::eq(lhs, rhs) || symbolic::uses(rhs, lhs)) {
                continue;
            }

            // Criterion: Write dominates read to not cause data races
            if (!users.dominates(*write, *read)) {
                continue;
            }

            // Collect all symbols used in the RHS
            std::unordered_set<std::string> rhs_symbols;
            for (auto& atom : symbolic::atoms(rhs)) {
                auto sym = SymEngine::rcp_static_cast<const SymEngine::Symbol>(atom);
                rhs_symbols.insert(sym->get_name());
            }

            // RHS' symbols may be written between write and read
            // We attempt to create the new RHS
            bool success = true;
            auto middle_users = users.all_uses_between(*write, *read);
            for (auto& user : middle_users) {
                if (user->use() != analysis::Use::WRITE) {
                    continue;
                }
                if (rhs_symbols.find(user->container()) == rhs_symbols.end()) {
                    continue;
                }

                success = false;
                break;
            }
            if (!success) {
                continue;
            }

            if (auto transition_stmt =
                    dynamic_cast<structured_control_flow::Transition*>(read->element())) {
                auto& assignments = transition_stmt->assignments();
                for (auto& entry : assignments) {
                    if (symbolic::uses(entry.second, lhs)) {
                        entry.second = symbolic::subs(entry.second, lhs, rhs);
                        applied = true;
                    }
                }
            } else if (auto if_else_stmt =
                           dynamic_cast<structured_control_flow::IfElse*>(read->element())) {
                // Criterion: RHS does not use nvptx symbols
                bool nvptx = false;
                for (auto& atom : symbolic::atoms(rhs)) {
                    auto sym = SymEngine::rcp_static_cast<const SymEngine::Symbol>(atom);
                    if (symbolic::is_nv(sym)) {
                        nvptx = true;
                        break;
                    }
                }
                if (nvptx) {
                    continue;
                }

                for (size_t i = 0; i < if_else_stmt->size(); i++) {
                    auto child = if_else_stmt->at(i);
                    if (symbolic::uses(child.second, lhs)) {
                        child.second = symbolic::subs(child.second, lhs, rhs);
                        applied = true;
                    }
                }
            } else if (auto memlet = dynamic_cast<data_flow::Memlet*>(read->element())) {
                bool used = false;
                for (auto& dim : memlet->subset()) {
                    if (symbolic::uses(dim, lhs)) {
                        dim = symbolic::subs(dim, lhs, rhs);
                        used = true;
                    }
                }
                if (used) {
                    applied = true;
                }
            } else if (auto tasklet = dynamic_cast<data_flow::Tasklet*>(read->element())) {
                auto& condition = tasklet->condition();
                if (symbolic::uses(condition, lhs)) {
                    tasklet->condition() = symbolic::subs(condition, lhs, rhs);
                    applied = true;
                }
            } else if (auto for_loop =
                           dynamic_cast<structured_control_flow::For*>(read->element())) {
                auto for_user = dynamic_cast<analysis::ForUser*>(read);
                if (for_user->is_init() && symbolic::uses(for_loop->init(), lhs)) {
                    for_loop->init() = symbolic::subs(for_loop->init(), lhs, rhs);
                    applied = true;
                } else if (for_user->is_condition() && symbolic::uses(for_loop->condition(), lhs)) {
                    for_loop->condition() = symbolic::subs(for_loop->condition(), lhs, rhs);
                    applied = true;
                } else if (for_user->is_update() && symbolic::uses(for_loop->update(), lhs)) {
                    for_loop->update() = symbolic::subs(for_loop->update(), lhs, rhs);
                    applied = true;
                }
            }
        }
    }

    return applied;
};

}  // namespace passes
}  // namespace sdfg
