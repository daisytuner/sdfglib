#include "sdfg/passes/symbolic/symbol_propagation.h"

#include "sdfg/analysis/data_dependency_analysis.h"

namespace sdfg {
namespace passes {

symbolic::Expression inverse(const symbolic::Symbol& lhs, const symbolic::Expression& rhs) {
    if (!symbolic::uses(rhs, lhs)) {
        return SymEngine::null;
    }

    if (SymEngine::is_a<SymEngine::Add>(*rhs)) {
        auto add = SymEngine::rcp_static_cast<const SymEngine::Add>(rhs);
        auto arg_0 = add->get_args()[0];
        auto arg_1 = add->get_args()[1];
        if (!symbolic::eq(arg_0, lhs)) {
            std::swap(arg_0, arg_1);
        }
        if (!symbolic::eq(arg_0, lhs)) {
            return SymEngine::null;
        }
        if (!SymEngine::is_a<SymEngine::Integer>(*arg_1)) {
            return SymEngine::null;
        }
        return symbolic::sub(lhs, arg_1);
    } else if (SymEngine::is_a<SymEngine::Mul>(*rhs)) {
        auto mul = SymEngine::rcp_static_cast<const SymEngine::Mul>(rhs);
        auto arg_0 = mul->get_args()[0];
        auto arg_1 = mul->get_args()[1];
        if (!symbolic::eq(arg_0, lhs)) {
            std::swap(arg_0, arg_1);
        }
        if (!symbolic::eq(arg_0, lhs)) {
            return SymEngine::null;
        }
        if (!SymEngine::is_a<SymEngine::Integer>(*arg_1)) {
            return SymEngine::null;
        }
        return symbolic::div(lhs, arg_1);
    }

    return SymEngine::null;
};

SymbolPropagation::SymbolPropagation()
    : Pass() {

      };

std::string SymbolPropagation::name() { return "SymbolPropagation"; };

bool SymbolPropagation::run_pass(builder::StructuredSDFGBuilder& builder,
                                 analysis::AnalysisManager& analysis_manager) {
    bool applied = false;

    auto& sdfg = builder.subject();
    auto& users = analysis_manager.get<analysis::Users>();
    auto& data_dependency_analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();
    for (auto& name : sdfg.containers()) {
        // Criterion: Only integers
        auto& type = builder.subject().type(name);
        auto scalar = dynamic_cast<const types::Scalar*>(&type);
        if (!scalar || !types::is_integer(scalar->primitive_type())) {
            continue;
        }

        // The symbol will become the LHS (to be replaced)
        auto lhs = symbolic::symbol(name);

        // Collect all reads of the symbol w.r.t to their writes
        auto raw_groups = data_dependency_analysis.defined_by(name);
        for (auto& entry : raw_groups) {
            // If not exclusive write, skip
            if (entry.second.size() != 1) {
                continue;
            }
            if (entry.first->use() != analysis::Use::READ) {
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
            auto rhs_modified = rhs;
            std::unordered_set<std::string> modified_symbols;

            auto middle_users = users.all_uses_between(*write, *read);
            for (auto& user : middle_users) {
                if (user->use() != analysis::Use::WRITE) {
                    continue;
                }
                if (rhs_symbols.find(user->container()) == rhs_symbols.end()) {
                    continue;
                }

                // Criterion: Symbol is only modified once
                if (modified_symbols.find(user->container()) != modified_symbols.end()) {
                    success = false;
                    break;
                }

                // Criterion: RHS must dominate modification
                if (!users.dominates(*write, *user)) {
                    success = false;
                    break;
                }

                // Criterion: Modification must dominate read
                if (!users.dominates(*user, *read)) {
                    success = false;
                    break;
                }

                // Criterion: Only transitions
                if (!dynamic_cast<structured_control_flow::Transition*>(user->element())) {
                    success = false;
                    break;
                }
                auto sym_transition =
                    dynamic_cast<structured_control_flow::Transition*>(user->element());
                auto sym_lhs = symbolic::symbol(user->container());
                auto sym_rhs = sym_transition->assignments().at(sym_lhs);

                // Limited to constants
                for (auto& atom : symbolic::atoms(sym_rhs)) {
                    if (!symbolic::eq(atom, sym_lhs)) {
                        success = false;
                        break;
                    }
                }
                if (!success) {
                    break;
                }

                auto inv = inverse(sym_lhs, sym_rhs);
                if (inv == SymEngine::null) {
                    success = false;
                    break;
                }

                rhs_modified = symbolic::subs(rhs_modified, sym_lhs, inv);
                modified_symbols.insert(user->container());
            }
            if (!success) {
                continue;
            }

            if (auto transition_stmt =
                    dynamic_cast<structured_control_flow::Transition*>(read->element())) {
                auto& assignments = transition_stmt->assignments();
                for (auto& entry : assignments) {
                    if (symbolic::uses(entry.second, lhs)) {
                        entry.second = symbolic::subs(entry.second, lhs, rhs_modified);
                        applied = true;
                    }
                }
            } else if (auto if_else_stmt =
                           dynamic_cast<structured_control_flow::IfElse*>(read->element())) {
                // Criterion: RHS does not use nvptx symbols
                bool nvptx = false;
                for (auto& atom : symbolic::atoms(rhs_modified)) {
                    if (symbolic::is_nv(atom)) {
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
                        child.second = symbolic::subs(child.second, lhs, rhs_modified);
                        applied = true;
                    }
                }
            } else if (auto memlet = dynamic_cast<data_flow::Memlet*>(read->element())) {
                bool used = false;
                for (auto& dim : memlet->subset()) {
                    if (symbolic::uses(dim, lhs)) {
                        dim = symbolic::subs(dim, lhs, rhs_modified);
                        used = true;
                    }
                }
                if (used) {
                    applied = true;
                }
            } else if (auto tasklet = dynamic_cast<data_flow::Tasklet*>(read->element())) {
                auto& condition = tasklet->condition();
                if (symbolic::uses(condition, lhs)) {
                    tasklet->condition() = symbolic::subs(condition, lhs, rhs_modified);
                    applied = true;
                }
            } else if (auto library_node = dynamic_cast<data_flow::LibraryNode*>(read->element())) {
                for (auto& symbol : library_node->symbols()) {
                    if (symbolic::eq(symbol, lhs)) {
                        library_node->replace(symbol, rhs_modified);
                        applied = true;
                    }
                }
            } else if (auto for_loop = dynamic_cast<structured_control_flow::StructuredLoop*>(
                           read->element())) {
                auto for_user = dynamic_cast<analysis::ForUser*>(read);
                if (for_user->is_init() && symbolic::uses(for_loop->init(), lhs)) {
                    for_loop->init() = symbolic::subs(for_loop->init(), lhs, rhs_modified);
                    applied = true;
                } else if (for_user->is_condition() && symbolic::uses(for_loop->condition(), lhs)) {
                    for_loop->condition() =
                        symbolic::subs(for_loop->condition(), lhs, rhs_modified);
                    applied = true;
                } else if (for_user->is_update() && symbolic::uses(for_loop->update(), lhs)) {
                    for_loop->update() = symbolic::subs(for_loop->update(), lhs, rhs_modified);
                    applied = true;
                }
            }
        }
    }

    return applied;
};

}  // namespace passes
}  // namespace sdfg
