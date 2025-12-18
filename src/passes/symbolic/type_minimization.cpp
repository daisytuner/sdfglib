#include "sdfg/passes/symbolic/type_minimization.h"

#include <unordered_set>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/symbolic/extreme_values.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace passes {

TypeMinimization::TypeMinimization(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
    : visitor::NonStoppingStructuredSDFGVisitor(builder, analysis_manager) {};

bool TypeMinimization::accept(structured_control_flow::Block& block) {
    bool applied = false;
    auto& dfg = block.dataflow();
    auto& assumptions_analysis = this->analysis_manager_.get<analysis::AssumptionsAnalysis>();
    auto& block_assumptions = assumptions_analysis.get(block, false);

    symbolic::ExpressionMap replacements;
    for (auto& edge : dfg.edges()) {
        auto& subset = edge.subset();
        for (auto& dim : subset) {
            auto truns = symbolic::find<SymEngine::FunctionSymbol>(dim);
            for (auto& trunc_func : truns) {
                if (!SymEngine::is_a<symbolic::TruncI32Function>(*trunc_func)) {
                    continue;
                }
                if (replacements.find(trunc_func) != replacements.end()) {
                    continue;
                }
                auto arg = trunc_func->get_args()[0];

                size_t output_bitwidth = 32;
                int64_t output_min_value_signed = 0;
                int64_t output_max_value_signed = (1ULL << (output_bitwidth - 1)) - 1;

                bool safe_trunc = true;
                auto mini = symbolic::minimum_new(arg, {}, block_assumptions, true);
                if (mini.is_null()) {
                    continue;
                }
                auto lb_criterion = symbolic::Ge(mini, symbolic::integer(output_min_value_signed));
                if (!symbolic::is_true(lb_criterion)) {
                    replacements[trunc_func] = SymEngine::null;
                    continue;
                }

                auto maxi = symbolic::maximum_new(arg, {}, block_assumptions, true);
                if (maxi.is_null()) {
                    continue;
                }
                auto ub_criterion = symbolic::Le(maxi, symbolic::integer(output_max_value_signed));
                if (!symbolic::is_true(ub_criterion)) {
                    replacements[trunc_func] = SymEngine::null;
                    continue;
                }

                replacements[trunc_func] = arg;
            }
        }
    }
    if (replacements.empty()) {
        return false;
    }

    for (auto& edge : dfg.edges()) {
        auto subset = edge.subset();
        for (auto& dim : subset) {
            for (auto& [old_expr, new_expr] : replacements) {
                if (new_expr.is_null()) {
                    continue;
                }
                dim = symbolic::subs(dim, old_expr, new_expr);
                applied = true;
            }
        }
        edge.set_subset(subset);
    }

    return applied;
}

} // namespace passes
} // namespace sdfg
