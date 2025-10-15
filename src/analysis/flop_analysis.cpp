#include "sdfg/analysis/flop_analysis.h"
#include <cassert>
#include <cstddef>
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

    for (size_t i = 0; i < sequence.size(); i++) {
        symbolic::Expression tmp = this->visit(sequence.at(i).first, analysis_manager);
        if (tmp.is_null()) return SymEngine::null;
        result = symbolic::add(result, tmp);
    }

    return result;
}

symbolic::Expression FlopAnalysis::visit_block(structured_control_flow::Block& block, AnalysisManager& analysis_manager) {
    symbolic::Expression result = symbolic::zero();
    auto& dfg = block.dataflow();

    for (auto tasklet : dfg.tasklets()) {
        if (tasklet->code() == data_flow::TaskletCode::fp_fma) {
            result = symbolic::add(result, symbolic::integer(2));
        } else if (data_flow::is_floating_point(tasklet->code())) {
            result = symbolic::add(result, symbolic::one());
        }
    }

    for (auto libnode : dfg.library_nodes()) {
        symbolic::Expression tmp = libnode->flop();
        if (tmp.is_null()) return SymEngine::null;
        result = symbolic::add(result, tmp);
    }

    return result;
}

symbolic::Expression FlopAnalysis::
    visit_structured_loop(structured_control_flow::StructuredLoop& loop, AnalysisManager& analysis_manager) {
    auto& loop_analysis = analysis_manager.get<LoopAnalysis>();
    auto& assumptions_analysis = analysis_manager.get<AssumptionsAnalysis>();
    auto canonical_bound = loop_analysis.canonical_bound(&loop, assumptions_analysis);

    auto indvar = loop.indvar();
    symbolic::SymbolVec symbols = {indvar};
    auto update_polynomial = symbolic::polynomial(loop.update(), symbols);
    auto update_coeffs = symbolic::affine_coefficients(update_polynomial, symbols);

    // For now, only allow polynomial of the form: 1 * indvar + n
    assert(update_coeffs.contains(indvar) && symbolic::eq(update_coeffs[indvar], symbolic::one()));
    symbolic::Expression stride = update_coeffs[symbolic::symbol("__daisy_constant__")];

    symbolic::Expression tmp = this->visit_sequence(loop.root(), analysis_manager);
    if (tmp.is_null()) return SymEngine::null;
    return symbolic::mul(symbolic::div(symbolic::sub(canonical_bound, loop.init()), stride), tmp);
}

symbolic::Expression FlopAnalysis::
    visit_if_else(structured_control_flow::IfElse& if_else, AnalysisManager& analysis_manager) {
    std::vector<symbolic::Expression> sub_flops;
    for (size_t i = 0; i < if_else.size(); i++) {
        symbolic::Expression tmp = this->visit_sequence(if_else.at(i).first, analysis_manager);
        if (tmp.is_null()) return SymEngine::null;
        sub_flops.push_back(tmp);
    }
    return SymEngine::max(sub_flops);
}

void FlopAnalysis::run(AnalysisManager& analysis_manager) {
    this->flop_ = this->visit_sequence(this->sdfg_.root(), analysis_manager);
}

FlopAnalysis::FlopAnalysis(StructuredSDFG& sdfg) : Analysis(sdfg), flop_(SymEngine::null) {}

symbolic::Expression FlopAnalysis::flop() { return this->flop_; }

} // namespace analysis
} // namespace sdfg
