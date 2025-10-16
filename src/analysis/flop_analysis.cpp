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
#include "symengine/symengine_rcp.h"

namespace sdfg {
namespace analysis {

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

    if (is_null) return SymEngine::null;
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
        if (tmp.is_null()) return SymEngine::null;
        libnodes_result = symbolic::add(libnodes_result, tmp);
    }

    // Filter the loop index variables in libnodes_result, and replace them by (upper_bound - lower_bound) / 2
    auto& assumptions_analysis = analysis_manager.get<AssumptionsAnalysis>();
    auto block_assumptions = assumptions_analysis.get(block);
    auto libnodes_result_atoms = symbolic::atoms(libnodes_result);
    for (auto sym : libnodes_result_atoms) {
        if (!block_assumptions.contains(sym)) continue;
        symbolic::Assumption assumption = block_assumptions.at(sym);
        if (!assumption.constant() || assumption.map().is_null()) continue;
        libnodes_result = symbolic::subs(
            libnodes_result,
            sym,
            symbolic::div(symbolic::sub(assumption.upper_bound(), assumption.lower_bound()), symbolic::integer(2))
        );
    }

    return symbolic::add(tasklets_result, libnodes_result);
}

symbolic::Expression FlopAnalysis::
    visit_structured_loop(structured_control_flow::StructuredLoop& loop, AnalysisManager& analysis_manager) {
    symbolic::Expression tmp = this->visit_sequence(loop.root(), analysis_manager);
    this->flops_[&loop.root()] = tmp;
    if (tmp.is_null()) return SymEngine::null;

    auto& assumptions_analysis = analysis_manager.get<AssumptionsAnalysis>();
    auto bound = LoopAnalysis::canonical_bound(&loop, assumptions_analysis);

    auto init = loop.init();

    auto indvar = loop.indvar();
    symbolic::SymbolVec symbols = {indvar};
    auto update_polynomial = symbolic::polynomial(loop.update(), symbols);
    auto update_coeffs = symbolic::affine_coefficients(update_polynomial, symbols);

    // For now, only allow polynomial of the form: 1 * indvar + n
    assert(update_coeffs.contains(indvar) && symbolic::eq(update_coeffs[indvar], symbolic::one()));
    symbolic::Expression stride = update_coeffs[symbolic::symbol("__daisy_constant__")];

    // Filter the loop index variables in bound, init, and stride, and replace them by (upper_bound - lower_bound) / 2
    auto loop_assumptions = assumptions_analysis.get(loop.root());
    auto bound_atoms = symbolic::atoms(bound);
    for (auto sym : bound_atoms) {
        if (!loop_assumptions.contains(sym)) continue;
        symbolic::Assumption assumption = loop_assumptions.at(sym);
        if (!assumption.constant() || assumption.map().is_null()) continue;
        bound = symbolic::subs(
            bound,
            sym,
            symbolic::div(symbolic::sub(assumption.upper_bound(), assumption.lower_bound()), symbolic::integer(2))
        );
    }
    auto init_atoms = symbolic::atoms(init);
    for (auto sym : init_atoms) {
        if (!loop_assumptions.contains(sym)) continue;
        symbolic::Assumption assumption = loop_assumptions.at(sym);
        if (!assumption.constant() || assumption.map().is_null()) continue;
        init = symbolic::subs(
            init,
            sym,
            symbolic::div(symbolic::sub(assumption.upper_bound(), assumption.lower_bound()), symbolic::integer(2))
        );
    }
    auto stride_atoms = symbolic::atoms(stride);
    for (auto sym : stride_atoms) {
        if (!loop_assumptions.contains(sym)) continue;
        symbolic::Assumption assumption = loop_assumptions.at(sym);
        if (!assumption.constant() || assumption.map().is_null()) continue;
        stride = symbolic::subs(
            stride,
            sym,
            symbolic::div(symbolic::sub(assumption.upper_bound(), assumption.lower_bound()), symbolic::integer(2))
        );
    }

    return symbolic::mul(symbolic::div(symbolic::sub(bound, init), stride), tmp);
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

    if (is_null) return SymEngine::null;
    return SymEngine::max(sub_flops);
}

symbolic::Expression FlopAnalysis::visit_while(structured_control_flow::While& loop, AnalysisManager& analysis_manager) {
    this->flops_[&loop.root()] = this->visit_sequence(loop.root(), analysis_manager);
    // Return null because there is now good way to simply estimate the FLOPs of a while loop
    return SymEngine::null;
}

void FlopAnalysis::run(AnalysisManager& analysis_manager) {
    this->flops_.clear();
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

} // namespace analysis
} // namespace sdfg
