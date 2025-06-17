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

    bool is_monotonic(structured_control_flow::StructuredLoop* loop) const;

    bool is_contiguous(structured_control_flow::StructuredLoop* loop) const;

    /**
     * @brief Describes the bound of a loop as a closed-form expression for contiguous loops.
     *
     * Example: i <= N && i < M -> i < min(N + 1, M)
     *
     * @param loop The loop to describe the bound of.
     * @return The bound of the loop as a closed-form expression, otherwise null.
     */
    symbolic::Expression canonical_bound(structured_control_flow::StructuredLoop* loop) const;
};

}  // namespace analysis
}  // namespace sdfg
