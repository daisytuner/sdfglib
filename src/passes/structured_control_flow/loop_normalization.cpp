#include "sdfg/passes/structured_control_flow/loop_normalization.h"

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/symbolic/conjunctive_normal_form.h"
#include "sdfg/symbolic/polynomials.h"
#include "sdfg/symbolic/series.h"

namespace sdfg {
namespace passes {

LoopNormalization::LoopNormalization(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
    : NonStoppingStructuredSDFGVisitor(builder, analysis_manager) {

      };

bool LoopNormalization::accept(structured_control_flow::For& loop) {
    bool applied = false;

    // Step 1: Bring condition to CNF
    symbolic::Condition condition = loop.condition();
    sdfg::symbolic::CNF cnf;
    try {
        cnf = symbolic::conjunctive_normal_form(condition);
    } catch (const symbolic::CNFException e) {
        return false;
    }
    // Update if changed
    {
        symbolic::Condition new_condition = symbolic::__true__();
        for (auto& clause : cnf) {
            symbolic::Condition new_clause = symbolic::__false__();
            for (auto& literal : clause) {
                auto new_literal = SymEngine::rcp_dynamic_cast<const SymEngine::Boolean>(symbolic::simplify(literal));
                new_clause = symbolic::Or(new_clause, new_literal);
            }
            new_condition = symbolic::And(new_condition, new_clause);
        }
        if (!symbolic::eq(new_condition, condition)) {
            builder_.update_loop(loop, loop.indvar(), new_condition, loop.init(), loop.update());
            applied = true;
        }
        condition = new_condition;
    }

    // Following steps require affine update
    auto indvar = loop.indvar();
    auto update = loop.update();
    auto& assumptions_analysis = analysis_manager_.get<analysis::AssumptionsAnalysis>();
    auto& assums = assumptions_analysis.get(loop, true);
    auto [success, coeffs] = symbolic::series::affine_int_coeffs(update, indvar, assums);
    if (!success) {
        return applied;
    }
    auto [mul_coeff, add_coeff] = coeffs;
    if (mul_coeff->as_int() != 1) {
        return applied;
    }
    int stride = add_coeff->as_int();
    if (stride == 0) {
        return applied;
    }

    // Step 2: Simplify literals of CNF that involve the induction variable
    symbolic::CNF new_cnf;
    for (auto& clause : cnf) {
        std::vector<symbolic::Condition> new_clause;
        for (auto& literal : clause) {
            if (SymEngine::is_a<SymEngine::Unequality>(*literal)) {
                auto old_literal = SymEngine::rcp_static_cast<const SymEngine::Unequality>(literal);
                auto lhs = old_literal->get_args().at(0);
                auto rhs = old_literal->get_args().at(1);
                if (symbolic::uses(lhs, indvar) && symbolic::uses(rhs, indvar)) {
                    new_clause.push_back(literal);
                    continue;
                }
                if (!symbolic::uses(rhs, indvar)) {
                    std::swap(lhs, rhs);
                }
                if (!symbolic::uses(rhs, indvar)) {
                    new_clause.push_back(literal);
                    continue;
                }

                // RHS is now guranteed to use the indvar

                // 1. Solve inequality for indvar (affine case)
                symbolic::SymbolVec syms = {indvar};
                auto poly_rhs = symbolic::polynomial(rhs, syms);
                if (poly_rhs == SymEngine::null) {
                    new_clause.push_back(literal);
                    continue;
                }
                auto coeffs_rhs = symbolic::affine_coefficients(poly_rhs, syms);
                auto mul_coeff_rhs = coeffs_rhs[indvar];
                auto add_coeff_rhs = coeffs_rhs[symbolic::symbol("__daisy_constant__")];

                auto new_rhs = symbolic::sub(lhs, add_coeff_rhs);
                // TODO: integer division
                new_rhs = symbolic::div(new_rhs, mul_coeff_rhs);
                auto new_lhs = indvar;

                // 2. Convert to comparison based on stride sign

                // Special cases: |stride| == 1
                if (stride == 1) {
                    auto new_literal = symbolic::Lt(new_lhs, new_rhs);
                    new_clause.push_back(new_literal);
                    continue;
                } else if (stride == -1) {
                    auto new_literal = symbolic::Gt(new_lhs, new_rhs);
                    new_clause.push_back(new_literal);
                    continue;
                }

                // Fallback general case
                new_clause.push_back(symbolic::Ne(new_lhs, new_rhs));
            } else if (SymEngine::is_a<SymEngine::StrictLessThan>(*literal)) {
                // Remove trivial max expressions
                auto slt = SymEngine::rcp_static_cast<const SymEngine::StrictLessThan>(literal);
                auto lhs = slt->get_args().at(0);
                auto rhs = slt->get_args().at(1);
                if (!symbolic::eq(lhs, indvar)) {
                    new_clause.push_back(literal);
                    continue;
                }
                if (!SymEngine::is_a<SymEngine::Max>(*rhs)) {
                    new_clause.push_back(literal);
                    continue;
                }
                auto max_expr = SymEngine::rcp_static_cast<const SymEngine::Max>(rhs);
                auto args = max_expr->get_args();
                if (args.size() != 2) {
                    new_clause.push_back(literal);
                    continue;
                }
                auto arg1 = args.at(0);
                auto arg2 = args.at(1);
                if (!symbolic::eq(arg1, loop.init())) {
                    std::swap(arg1, arg2);
                }
                if (!symbolic::eq(arg1, loop.init())) {
                    new_clause.push_back(literal);
                    continue;
                }
                auto new_literal = symbolic::Lt(indvar, arg2);
                new_clause.push_back(new_literal);
            } else {
                new_clause.push_back(literal);
            }
        }
        new_cnf.push_back(new_clause);
    }
    // Update if changed
    {
        symbolic::Condition new_condition = symbolic::__true__();
        for (auto& clause : new_cnf) {
            symbolic::Condition new_clause = symbolic::__false__();
            for (auto& literal : clause) {
                new_clause = symbolic::Or(new_clause, literal);
            }
            new_condition = symbolic::And(new_condition, new_clause);
        }
        if (!symbolic::eq(new_condition, condition)) {
            builder_.update_loop(loop, loop.indvar(), new_condition, loop.init(), loop.update());
            applied = true;
            condition = new_condition;
        }
    }

    // Check uses of indvar in loop body are all memlets for further steps
    auto& users_analysis = analysis_manager_.get<analysis::Users>();
    analysis::UsersView body_users(users_analysis, loop.root());
    bool all_subsets = true;
    for (auto& user : body_users.uses(indvar->get_name())) {
        if (!dynamic_cast<data_flow::Memlet*>(user->element())) {
            all_subsets = false;
            break;
        }
    }
    if (!all_subsets) {
        return applied;
    }

    try {
        new_cnf = symbolic::conjunctive_normal_form(condition);
    } catch (const symbolic::CNFException e) {
        return applied;
    }

    if (stride > 0) {
        // Step 3: Shift loop to start from zero if possible
        if (!symbolic::eq(loop.init(), symbolic::zero())) {
            // Require integer initial value for following steps
            if (!SymEngine::is_a<SymEngine::Integer>(*loop.init())) {
                return applied;
            }

            auto new_init = symbolic::zero();
            auto actual_indvar = symbolic::add(indvar, loop.init());

            // T
            bool canonical_condition = false;
            if (new_cnf.size() == 1) {
                auto& clause = new_cnf[0];
                if (clause.size() == 1) {
                    auto literal = clause[0];
                    if (SymEngine::is_a<SymEngine::StrictLessThan>(*literal)) {
                        auto lhs = literal->get_args().at(0);
                        auto rhs = literal->get_args().at(1);
                        if (symbolic::eq(lhs, indvar) && !symbolic::uses(rhs, indvar)) {
                            condition = symbolic::Lt(indvar, symbolic::sub(rhs, loop.init()));
                            canonical_condition = true;
                        }
                    }
                }
            }
            if (!canonical_condition) {
                condition = symbolic::subs(condition, indvar, actual_indvar);
            }
            builder_.update_loop(loop, indvar, condition, new_init, loop.update());
            auto& root = loop.root();
            root.replace(indvar, actual_indvar);

            return true;
        }
    } else if (stride < 0) {
        // Step 4: Rotate loop if stride is negative
        if (stride != -1) {
            return applied;
        }
        if (new_cnf.size() != 1) {
            return applied;
        }
        auto& clause = new_cnf[0];
        if (clause.size() != 1) {
            return applied;
        }
        auto literal = clause[0];
        if (!SymEngine::is_a<SymEngine::StrictLessThan>(*literal)) {
            return applied;
        }
        auto slt = SymEngine::rcp_static_cast<const SymEngine::StrictLessThan>(literal);
        auto lhs = slt->get_args().at(0);
        auto rhs = slt->get_args().at(1);
        if (!symbolic::eq(rhs, indvar)) {
            return applied;
        }

        // Update loop parameters
        auto new_init = symbolic::add(lhs, symbolic::one());
        auto new_update = symbolic::add(indvar, symbolic::one());
        auto new_condition = symbolic::Lt(indvar, symbolic::add(loop.init(), symbolic::one()));

        // Replace indvar by (init - indvar) in loop body
        loop.root().replace(indvar, symbolic::sub(loop.init(), symbolic::sub(indvar, new_init)));

        builder_.update_loop(loop, loop.indvar(), new_condition, new_init, new_update);
        return true;
    }

    return applied;
};

} // namespace passes
} // namespace sdfg
