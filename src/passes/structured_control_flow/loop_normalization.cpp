#include "sdfg/passes/structured_control_flow/loop_normalization.h"

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/symbolic/conjunctive_normal_form.h"
#include "sdfg/symbolic/series.h"

namespace sdfg {
namespace passes {

bool LoopNormalization::apply(builder::StructuredSDFGBuilder& builder,
                              analysis::AnalysisManager& analysis_manager,
                              structured_control_flow::For& loop) {
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    if (!loop_analysis.is_contiguous(&loop)) {
        return false;
    }

    // Section: Condition
    // Turn inequalities into strict less than
    auto indvar = loop.indvar();
    auto condition = loop.condition();

    bool applied = false;
    try {
        auto cnf = symbolic::conjunctive_normal_form(condition);
        symbolic::CNF new_cnf;
        for (auto& clause : cnf) {
            std::vector<symbolic::Condition> new_clause;
            for (auto& literal : clause) {
                if (SymEngine::is_a<SymEngine::Unequality>(*literal)) {
                    auto eq = SymEngine::rcp_static_cast<const SymEngine::Unequality>(literal);
                    auto eq_args = eq->get_args();
                    auto lhs = eq_args.at(0);
                    auto rhs = eq_args.at(1);
                    if (SymEngine::eq(*lhs, *indvar) && !symbolic::uses(rhs, indvar)) {
                        new_clause.push_back(symbolic::Lt(lhs, rhs));
                        applied = true;
                    } else if (SymEngine::eq(*rhs, *indvar) && !symbolic::uses(lhs, indvar)) {
                        new_clause.push_back(symbolic::Lt(rhs, lhs));
                        applied = true;
                    } else {
                        new_clause.push_back(literal);
                    }
                } else {
                    new_clause.push_back(literal);
                }
            }
            new_cnf.push_back(new_clause);
        }
        // Construct new condition
        symbolic::Condition new_condition = symbolic::__true__();
        for (auto& clause : new_cnf) {
            symbolic::Condition new_clause = symbolic::__false__();
            for (auto& literal : clause) {
                new_clause = symbolic::Or(new_clause, literal);
            }
            new_condition = symbolic::And(new_condition, new_clause);
        }
        loop.condition() = new_condition;
    } catch (const symbolic::CNFException e) {
    }

    return applied;
};

LoopNormalization::LoopNormalization()
    : Pass() {

      };

std::string LoopNormalization::name() { return "LoopNormalization"; };

bool LoopNormalization::run_pass(builder::StructuredSDFGBuilder& builder,
                                 analysis::AnalysisManager& analysis_manager) {
    bool applied = false;

    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    for (const auto& loop : loop_analysis.loops()) {
        if (auto for_loop = dynamic_cast<structured_control_flow::For*>(loop)) {
            applied |= this->apply(builder, analysis_manager, *for_loop);
        }
    }

    return applied;
};

}  // namespace passes
}  // namespace sdfg
