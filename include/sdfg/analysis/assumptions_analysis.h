#pragma once

#include <unordered_map>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/symbolic/assumptions.h"
#include "sdfg/symbolic/conjunctive_normal_form.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace analysis {

class AssumptionsAnalysis : public Analysis {
private:
    std::unordered_map<structured_control_flow::ControlFlowNode*, symbolic::Assumptions> assumptions_;

    symbolic::SymbolSet parameters_;

    analysis::ScopeAnalysis* scope_analysis_;

    analysis::Users* users_analysis_;

    void traverse(structured_control_flow::Sequence& root, analysis::AnalysisManager& analysis_manager);

    void visit_block(structured_control_flow::Block* block, analysis::AnalysisManager& analysis_manager);

    void visit_sequence(structured_control_flow::Sequence* sequence, analysis::AnalysisManager& analysis_manager);

    void visit_if_else(structured_control_flow::IfElse* if_else, analysis::AnalysisManager& analysis_manager);

    void visit_while(structured_control_flow::While* while_loop, analysis::AnalysisManager& analysis_manager);

    void visit_structured_loop(structured_control_flow::StructuredLoop* loop, analysis::AnalysisManager& analysis_manager);

    void determine_parameters(analysis::AnalysisManager& analysis_manager);

protected:
    void run(analysis::AnalysisManager& analysis_manager) override;

public:
    AssumptionsAnalysis(StructuredSDFG& sdfg);

    const symbolic::Assumptions get(structured_control_flow::ControlFlowNode& node, bool include_trivial_bounds = false);

    const symbolic::Assumptions
    get(structured_control_flow::ControlFlowNode& from,
        structured_control_flow::ControlFlowNode& to,
        bool include_trivial_bounds = false);

    const symbolic::SymbolSet& parameters();

    bool is_parameter(const symbolic::Symbol& container);

    bool is_parameter(const std::string& container);

    void add(symbolic::Assumptions& assumptions, structured_control_flow::ControlFlowNode& node);

    static symbolic::Expression cnf_to_upper_bound(const symbolic::CNF& cnf, const symbolic::Symbol indvar);
};

} // namespace analysis
} // namespace sdfg
