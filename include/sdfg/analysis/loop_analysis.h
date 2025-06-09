#pragma once

#include "sdfg/analysis/analysis.h"

namespace sdfg {
namespace analysis {

class LoopAnalysis : public Analysis {
   private:
    std::unordered_set<structured_control_flow::ControlFlowNode*> loops_;
    std::unordered_map<structured_control_flow::ControlFlowNode*,
                       structured_control_flow::ControlFlowNode*>
        loop_tree_;

    void run(structured_control_flow::ControlFlowNode& scope,
             structured_control_flow::ControlFlowNode* parent_loop);

   protected:
    void run(analysis::AnalysisManager& analysis_manager) override;

   public:
    LoopAnalysis(StructuredSDFG& sdfg);

    const std::unordered_set<structured_control_flow::ControlFlowNode*> loops() const;

    const std::unordered_map<structured_control_flow::ControlFlowNode*,
                             structured_control_flow::ControlFlowNode*>&
    loop_tree() const;

    structured_control_flow::ControlFlowNode* parent_loop(
        structured_control_flow::ControlFlowNode* loop) const;

    const std::vector<structured_control_flow::ControlFlowNode*> outermost_loops() const;

    bool is_monotonic(const structured_control_flow::StructuredLoop* loop) const;

    bool is_contiguous(const structured_control_flow::StructuredLoop* loop) const;
};

}  // namespace analysis
}  // namespace sdfg
