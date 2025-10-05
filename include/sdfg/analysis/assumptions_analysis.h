#pragma once

#include <set>
#include <unordered_map>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/symbolic/assumptions.h"
#include "sdfg/symbolic/conjunctive_normal_form.h"

namespace sdfg {
namespace analysis {

class AssumptionsAnalysis : public Analysis {
private:
    std::unordered_map<structured_control_flow::ControlFlowNode*, symbolic::Assumptions> assumptions_;

    void traverse(structured_control_flow::Sequence& root, analysis::AnalysisManager& analysis_manager);

    void visit_block(structured_control_flow::Block* block, analysis::AnalysisManager& analysis_manager);

    void visit_sequence(structured_control_flow::Sequence* sequence, analysis::AnalysisManager& analysis_manager);

    void visit_if_else(structured_control_flow::IfElse* if_else, analysis::AnalysisManager& analysis_manager);

    void visit_while(structured_control_flow::While* while_loop, analysis::AnalysisManager& analysis_manager);

    void visit_for(structured_control_flow::For* for_loop, analysis::AnalysisManager& analysis_manager);

    void visit_map(structured_control_flow::Map* map, analysis::AnalysisManager& analysis_manager);

protected:
    void run(analysis::AnalysisManager& analysis_manager) override;

public:
    AssumptionsAnalysis(StructuredSDFG& sdfg);

    const symbolic::Assumptions get(structured_control_flow::ControlFlowNode& node, bool include_trivial_bounds = false);

    const symbolic::Assumptions
    get(structured_control_flow::ControlFlowNode& from,
        structured_control_flow::ControlFlowNode& to,
        bool include_trivial_bounds = false);

    void add(symbolic::Assumptions& assumptions, structured_control_flow::ControlFlowNode& node);

    static symbolic::Expression cnf_to_upper_bound(const symbolic::CNF& cnf, const symbolic::Symbol indvar);
};

} // namespace analysis
} // namespace sdfg
