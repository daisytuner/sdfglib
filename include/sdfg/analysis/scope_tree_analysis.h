#pragma once

#include "sdfg/analysis/analysis.h"

namespace sdfg {
namespace analysis {

class ScopeTreeAnalysis : public Analysis {
   private:
    std::unordered_map<structured_control_flow::ControlFlowNode*,
                       structured_control_flow::ControlFlowNode*>
        scope_tree_;

    void run(structured_control_flow::ControlFlowNode* current,
             structured_control_flow::ControlFlowNode* parent_scope);

   protected:
    void run(analysis::AnalysisManager& analysis_manager) override;

   public:
    ScopeTreeAnalysis(StructuredSDFG& sdfg);

    const std::unordered_map<structured_control_flow::ControlFlowNode*,
                             structured_control_flow::ControlFlowNode*>&
    scope_tree() const;

    structured_control_flow::ControlFlowNode* parent_scope(
        structured_control_flow::ControlFlowNode* scope) const;
};

}  // namespace analysis
}  // namespace sdfg
