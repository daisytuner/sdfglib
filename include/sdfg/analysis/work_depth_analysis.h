#pragma once

#include "sdfg/analysis/analysis.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace analysis {

class WorkDepthAnalysis : public Analysis {
private:
    std::unordered_map<const structured_control_flow::ControlFlowNode*, symbolic::Expression> work_;
    std::unordered_map<const structured_control_flow::ControlFlowNode*, symbolic::Expression> depth_;

    symbolic::SymbolSet while_symbols_;

protected:
    void run(analysis::AnalysisManager& analysis_manager) override;

public:
    WorkDepthAnalysis(StructuredSDFG& sdfg);

    const symbolic::Expression& work(const structured_control_flow::ControlFlowNode* node) const;

    const symbolic::Expression& depth(const structured_control_flow::ControlFlowNode* node) const;

    const symbolic::SymbolSet while_symbols(symbolic::Expression expression) const;
};

} // namespace analysis
} // namespace sdfg
