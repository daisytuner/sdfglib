#include "sdfg/passes/structured_control_flow/loop_normalization.h"

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/symbolic/conjunctive_normal_form.h"
#include "sdfg/symbolic/polynomials.h"
#include "sdfg/symbolic/series.h"

namespace sdfg {
namespace passes {

bool LoopNormalization::apply(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::For& loop
) {
    bool applied = false;

    // Step 1: Bring condition to CNF
    auto condition = loop.condition();
    sdfg::symbolic::CNF cnf;
    try {
        cnf = symbolic::conjunctive_normal_form(condition);
    } catch (const symbolic::CNFException e) {
        return false;
    }
    {
        symbolic::Condition new_condition = symbolic::__true__();
        for (auto& clause : cnf) {
            symbolic::Condition new_clause = symbolic::__false__();
            for (auto& literal : clause) {
                new_clause = symbolic::Or(new_clause, literal);
            }
            new_condition = symbolic::And(new_condition, new_clause);
        }
        if (!symbolic::eq(new_condition, condition)) {
            builder.update_loop(loop, loop.indvar(), new_condition, loop.init(), loop.update());
            applied = true;
        }
        condition = new_condition;
    }

    // Step 2: Normalize bound
    auto indvar = loop.indvar();
    auto update = loop.update();

    // check if update is affine in indvar
    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto assums = assumptions_analysis.get(loop, true);
    auto [success, coeffs] = symbolic::series::affine_int_coeffs(update, indvar, assums);
    if (!success) {
        return applied;
    }
    auto [mul_coeff, add_coeff] = coeffs;
    if (mul_coeff->as_int() != 1) {
        return applied;
    }
    int stride = add_coeff->as_int();

    // Convert inequality-literals into comparisons with loop variable on LHS
    symbolic::CNF new_cnf;
    for (auto& clause : cnf) {
        std::vector<symbolic::Condition> new_clause;
        for (auto& literal : clause) {
            if (!SymEngine::is_a<SymEngine::Unequality>(*literal)) {
                new_clause.push_back(literal);
                continue;
            }
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

            // TODO: Modulo case: stride != +/-1
            new_clause.push_back(symbolic::Ne(new_lhs, new_rhs));
        }
        new_cnf.push_back(new_clause);
    }
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
            builder.update_loop(loop, loop.indvar(), new_condition, loop.init(), loop.update());
            applied = true;
        }
    }

    // Step 3: Rotate loop if stride is negative
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

    builder.update_loop(loop, loop.indvar(), new_condition, new_init, new_update);

    return applied;
};

LoopNormalization::LoopNormalization()
    : Pass() {

      };

std::string LoopNormalization::name() { return "LoopNormalization"; };

bool LoopNormalization::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    bool applied = false;

    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    for (const auto& loop : loop_analysis.loops()) {
        if (auto for_loop = dynamic_cast<structured_control_flow::For*>(loop)) {
            applied |= this->apply(builder, analysis_manager, *for_loop);
        }
    }

    return applied;
};

} // namespace passes
} // namespace sdfg
