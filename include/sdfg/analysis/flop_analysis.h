#pragma once

#include "sdfg/analysis/analysis.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace analysis {

class FlopAnalysis : public Analysis {
private:
    symbolic::Expression flop_;

    symbolic::Expression visit(structured_control_flow::ControlFlowNode& node, AnalysisManager& analysis_manager);

    symbolic::Expression visit_sequence(structured_control_flow::Sequence& sequence, AnalysisManager& analysis_manager);

    symbolic::Expression visit_block(structured_control_flow::Block& block, AnalysisManager& analysis_manager);

    symbolic::Expression
    visit_structured_loop(structured_control_flow::StructuredLoop& loop, AnalysisManager& analysis_manager);

    symbolic::Expression visit_if_else(structured_control_flow::IfElse& if_else, AnalysisManager& analysis_manager);

protected:
    void run(AnalysisManager& analysis_manager) override;

public:
    FlopAnalysis(StructuredSDFG& sdfg);

    symbolic::Expression flop();
};

} // namespace analysis
} // namespace sdfg
