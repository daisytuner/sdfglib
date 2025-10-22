#include "sdfg/analysis/flop_analysis.h"
#include <cassert>
#include <cstddef>
#include <unordered_map>
#include <vector>
#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/return.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/structured_control_flow/while.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/symbolic/assumptions.h"
#include "sdfg/symbolic/polynomials.h"
#include "sdfg/symbolic/symbolic.h"
#include "symengine/functions.h"
#include "symengine/symengine_rcp.h"

namespace sdfg {
namespace analysis {

/// An expression is a parameter expression if all its symbols are parameters
bool FlopAnalysis::is_parameter_expression(const symbolic::Expression& expr) {
    if (expr.is_null()) {
        return false;
    }
    for (auto& sym : symbolic::atoms(expr)) {
        if (!this->parameters_.contains(sym)) {
            return false;
        }
    }
    return true;
}

symbolic::ExpressionSet FlopAnalysis::choose_bounds(const symbolic::ExpressionSet& bounds) {
    symbolic::ExpressionSet result;
    for (auto& bound : bounds) {
        if (symbolic::eq(bound, SymEngine::NegInf) || symbolic::eq(bound, SymEngine::Inf)) {
            // Skip infinities
            continue;
        } else if (SymEngine::is_a<SymEngine::Integer>(*bound)) {
            // Collect integers
            result.insert(bound);
        } else if (!symbolic::contains_dynamic_sizeof(bound) && this->is_parameter_expression(bound)) {
            // Collect parameter expressions if they do not contain dynamic_sizeof
            result.insert(bound);
        }
    }
    if (result.empty()) {
        // Fallback if no integers or parameter expressions were found
        return bounds;
    } else {
        return result;
    }
}

symbolic::Expression FlopAnalysis::
    replace_loop_indices(const symbolic::Expression expr, symbolic::Assumptions& assumptions) {
    symbolic::Expression result = expr;
    auto atoms = symbolic::atoms(result);
    for (auto sym : atoms) {
        if (!assumptions.contains(sym)) continue;
        symbolic::Assumption assumption = assumptions.at(sym);
        if (!assumption.constant() || assumption.map().is_null()) continue;
        symbolic::Expression ub, lb;
        if (assumption.tight_upper_bound().is_null()) {
            auto bounds = this->choose_bounds(assumption.upper_bounds());
            if (bounds.empty()) {
                ub = assumption.upper_bound();
            } else {
                ub = SymEngine::min(std::vector<symbolic::Expression>(bounds.begin(), bounds.end()));
            }
        } else {
            ub = assumption.tight_upper_bound();
        }
        if (assumption.tight_lower_bound().is_null()) {
            auto bounds = this->choose_bounds(assumption.lower_bounds());
            if (bounds.empty()) {
                lb = assumption.lower_bound();
            } else {
                lb = SymEngine::max(std::vector<symbolic::Expression>(bounds.begin(), bounds.end()));
            }
        } else {
            lb = assumption.tight_lower_bound();
        }
        result = symbolic::subs(result, sym, symbolic::div(symbolic::sub(ub, lb), symbolic::integer(2)));
        this->precise_ = false;
    }
    return result;
}

symbolic::Expression FlopAnalysis::visit(structured_control_flow::ControlFlowNode& node, AnalysisManager& analysis_manager) {
    if (auto sequence = dynamic_cast<structured_control_flow::Sequence*>(&node)) {
        return this->visit_sequence(*sequence, analysis_manager);
    } else if (auto block = dynamic_cast<structured_control_flow::Block*>(&node)) {
        return this->visit_block(*block, analysis_manager);
    } else if (auto structured_loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&node)) {
        return this->visit_structured_loop(*structured_loop, analysis_manager);
    } else if (auto if_else = dynamic_cast<structured_control_flow::IfElse*>(&node)) {
        return this->visit_if_else(*if_else, analysis_manager);
    } else if (auto while_loop = dynamic_cast<structured_control_flow::While*>(&node)) {
        return this->visit_while(*while_loop, analysis_manager);
    } else if (dynamic_cast<structured_control_flow::Return*>(&node)) {
        return symbolic::zero();
    } else if (dynamic_cast<structured_control_flow::Break*>(&node)) {
        return symbolic::zero();
    } else if (dynamic_cast<structured_control_flow::Continue*>(&node)) {
        return symbolic::zero();
    } else {
        return SymEngine::null;
        this->precise_ = false;
    }
}

symbolic::Expression FlopAnalysis::
    visit_sequence(structured_control_flow::Sequence& sequence, AnalysisManager& analysis_manager) {
    symbolic::Expression result = symbolic::zero();
    bool is_null = false;

    for (size_t i = 0; i < sequence.size(); i++) {
        symbolic::Expression tmp = this->visit(sequence.at(i).first, analysis_manager);
        this->flops_[&sequence.at(i).first] = tmp;
        if (tmp.is_null()) is_null = true;
        if (!is_null) result = symbolic::add(result, tmp);
    }

    if (is_null) {
        this->precise_ = false;
        return SymEngine::null;
    }
    return result;
}

symbolic::Expression FlopAnalysis::visit_block(structured_control_flow::Block& block, AnalysisManager& analysis_manager) {
    auto& dfg = block.dataflow();

    symbolic::Expression tasklets_result = symbolic::zero();
    for (auto tasklet : dfg.tasklets()) {
        if (tasklet->code() == data_flow::TaskletCode::fp_fma) {
            tasklets_result = symbolic::add(tasklets_result, symbolic::integer(2));
        } else if (data_flow::is_floating_point(tasklet->code())) {
            tasklets_result = symbolic::add(tasklets_result, symbolic::one());
        }
    }

    symbolic::Expression libnodes_result = symbolic::zero();
    for (auto libnode : dfg.library_nodes()) {
        symbolic::Expression tmp = libnode->flop();
        if (tmp.is_null()) {
            this->precise_ = false;
            return SymEngine::null;
        }
        libnodes_result = symbolic::add(libnodes_result, tmp);
    }

    // Filter the loop index variables in libnodes_result, and replace them by (upper_bound - lower_bound) / 2
    auto& assumptions_analysis = analysis_manager.get<AssumptionsAnalysis>();
    auto block_assumptions = assumptions_analysis.get(block);
    libnodes_result = this->replace_loop_indices(libnodes_result, block_assumptions);

    return symbolic::add(tasklets_result, libnodes_result);
}

symbolic::Expression FlopAnalysis::
    visit_structured_loop(structured_control_flow::StructuredLoop& loop, AnalysisManager& analysis_manager) {
    symbolic::Expression tmp = this->visit_sequence(loop.root(), analysis_manager);
    this->flops_[&loop.root()] = tmp;
    if (tmp.is_null()) {
        this->precise_ = false;
        return SymEngine::null;
    }

    // Require existance of assumptions for the loop indvar
    auto indvar = loop.indvar();
    auto& assumptions_analysis = analysis_manager.get<AssumptionsAnalysis>();
    auto loop_assumptions = assumptions_analysis.get(loop.root());
    if (!loop_assumptions.contains(indvar)) {
        this->precise_ = false;
        return SymEngine::null;
    }
    bool done;

    // Determine initial value of loop
    symbolic::Expression init = SymEngine::null;
    done = false;
    if (!loop_assumptions[indvar].tight_lower_bound().is_null()) {
        init = this->replace_loop_indices(loop_assumptions[indvar].tight_lower_bound(), loop_assumptions);
        done = this->is_parameter_expression(init);
    }
    if (!done && !symbolic::eq(loop_assumptions[indvar].lower_bound(), SymEngine::NegInf)) {
        auto bounds = this->choose_bounds(loop_assumptions[indvar].lower_bounds());
        if (!bounds.empty()) {
            init = this->replace_loop_indices(
                SymEngine::max(std::vector<symbolic::Expression>(bounds.begin(), bounds.end())), loop_assumptions
            );
            this->precise_ = false;
            done = this->is_parameter_expression(init);
        }
    }
    if (!done) {
        init = this->replace_loop_indices(loop.init(), loop_assumptions);
        this->precise_ = false;
    }
    if (init.is_null()) {
        this->precise_ = false;
        return SymEngine::null;
    }

    // Determine bound of loop
    symbolic::Expression bound;
    done = false;
    if (!loop_assumptions[indvar].tight_upper_bound().is_null()) {
        bound = this->replace_loop_indices(loop_assumptions[indvar].tight_upper_bound(), loop_assumptions);
        done = this->is_parameter_expression(bound);
    }
    if (!done && !symbolic::eq(loop_assumptions[indvar].upper_bound(), SymEngine::Inf)) {
        auto bounds = this->choose_bounds(loop_assumptions[indvar].upper_bounds());
        if (!bounds.empty()) {
            bound = this->replace_loop_indices(
                SymEngine::min(std::vector<symbolic::Expression>(bounds.begin(), bounds.end())), loop_assumptions
            );
            this->precise_ = false;
            done = this->is_parameter_expression(bound);
        }
    }
    if (!done) {
        auto canonical_bound = LoopAnalysis::canonical_bound(&loop, assumptions_analysis);
        if (!canonical_bound.is_null()) {
            bound = this->replace_loop_indices(symbolic::sub(canonical_bound, symbolic::one()), loop_assumptions);
            this->precise_ = false;
        }
    }
    if (bound.is_null()) {
        this->precise_ = false;
        return SymEngine::null;
    }

    // Determine stride of loop
    symbolic::SymbolVec symbols = {indvar};
    auto update_polynomial = symbolic::polynomial(loop.update(), symbols);
    if (update_polynomial.is_null()) {
        this->precise_ = false;
        return SymEngine::null;
    }
    auto update_coeffs = symbolic::affine_coefficients(update_polynomial, symbols);

    // For now, only allow polynomial of the form: 1 * indvar + n
    assert(update_coeffs.contains(indvar) && symbolic::eq(update_coeffs[indvar], symbolic::one()));
    symbolic::Expression stride =
        this->replace_loop_indices(update_coeffs[symbolic::symbol("__daisy_constant__")], loop_assumptions);

    return symbolic::mul(symbolic::div(symbolic::add(symbolic::sub(bound, init), symbolic::one()), stride), tmp);
}

symbolic::Expression FlopAnalysis::
    visit_if_else(structured_control_flow::IfElse& if_else, AnalysisManager& analysis_manager) {
    if (if_else.size() == 0) return symbolic::zero();

    std::vector<symbolic::Expression> sub_flops;
    bool is_null = false;

    for (size_t i = 0; i < if_else.size(); i++) {
        symbolic::Expression tmp = this->visit_sequence(if_else.at(i).first, analysis_manager);
        this->flops_[&if_else.at(i).first] = tmp;
        if (tmp.is_null()) is_null = true;
        if (!is_null) sub_flops.push_back(tmp);
    }

    this->precise_ = false;
    if (is_null) {
        return SymEngine::null;
    }
    return SymEngine::max(sub_flops);
}

symbolic::Expression FlopAnalysis::visit_while(structured_control_flow::While& loop, AnalysisManager& analysis_manager) {
    this->flops_[&loop.root()] = this->visit_sequence(loop.root(), analysis_manager);
    this->precise_ = false;
    // Return null because there is now good way to simply estimate the FLOPs of a while loop
    return SymEngine::null;
}

void FlopAnalysis::run(AnalysisManager& analysis_manager) {
    this->flops_.clear();
    this->precise_ = true;

    auto& assumptions_analysis = analysis_manager.get<AssumptionsAnalysis>();
    this->parameters_ = assumptions_analysis.parameters();

    this->flops_[&this->sdfg_.root()] = this->visit_sequence(this->sdfg_.root(), analysis_manager);
}

FlopAnalysis::FlopAnalysis(StructuredSDFG& sdfg) : Analysis(sdfg) {}

bool FlopAnalysis::contains(const structured_control_flow::ControlFlowNode* node) {
    return this->flops_.contains(node);
}

symbolic::Expression FlopAnalysis::get(const structured_control_flow::ControlFlowNode* node) {
    return this->flops_[node];
}

std::unordered_map<const structured_control_flow::ControlFlowNode*, symbolic::Expression> FlopAnalysis::get() {
    return this->flops_;
}

bool FlopAnalysis::precise() { return this->precise_; }

} // namespace analysis
} // namespace sdfg
