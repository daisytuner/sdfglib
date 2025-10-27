#pragma once

#include <unordered_map>
#include "sdfg/analysis/analysis.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/symbolic/assumptions.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace analysis {

class FlopAnalysis : public Analysis {
private:
    std::unordered_map<const structured_control_flow::ControlFlowNode*, symbolic::Expression> flops_;

    bool precise_;

    bool is_parameter_expression(const symbolic::SymbolSet& parameters, const symbolic::Expression& expr);

    symbolic::ExpressionSet choose_bounds(const symbolic::SymbolSet& parameters, const symbolic::ExpressionSet& bounds);

    symbolic::Expression replace_loop_indices(
        const symbolic::SymbolSet& parameters, const symbolic::Expression expr, symbolic::Assumptions& assumptions
    );

    symbolic::SymbolSet
    get_scope_parameters(const structured_control_flow::ControlFlowNode& scope, AnalysisManager& analysis_manager);

    symbolic::Expression visit(structured_control_flow::ControlFlowNode& node, AnalysisManager& analysis_manager);

    symbolic::Expression visit_sequence(structured_control_flow::Sequence& sequence, AnalysisManager& analysis_manager);

    symbolic::Expression visit_block(structured_control_flow::Block& block, AnalysisManager& analysis_manager);

    symbolic::Expression
    visit_structured_loop(structured_control_flow::StructuredLoop& loop, AnalysisManager& analysis_manager);

    symbolic::Expression visit_structured_loop_with_scope(
        structured_control_flow::StructuredLoop& loop,
        AnalysisManager& analysis_manager,
        structured_control_flow::ControlFlowNode& scope,
        symbolic::Expression child_expr
    );

    symbolic::Expression visit_if_else(structured_control_flow::IfElse& if_else, AnalysisManager& analysis_manager);

    symbolic::Expression visit_while(structured_control_flow::While& loop, AnalysisManager& analysis_manager);

protected:
    void run(AnalysisManager& analysis_manager) override;

public:
    FlopAnalysis(StructuredSDFG& sdfg);

    bool contains(const structured_control_flow::ControlFlowNode* node);

    symbolic::Expression get(const structured_control_flow::ControlFlowNode* node);

    std::unordered_map<const structured_control_flow::ControlFlowNode*, symbolic::Expression> get();

    bool precise();
};

} // namespace analysis
} // namespace sdfg
