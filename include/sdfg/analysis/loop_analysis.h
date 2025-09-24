#pragma once

#include <unordered_set>
#include "sdfg/analysis/analysis.h"
#include "sdfg/structured_control_flow/structured_loop.h"

namespace sdfg {
namespace analysis {

class AssumptionsAnalysis;

struct DFSLoopComparator {
    const std::vector<structured_control_flow::ControlFlowNode*>* loops_order_;

    DFSLoopComparator(const std::vector<structured_control_flow::ControlFlowNode*>* loops_order)
        : loops_order_(loops_order) {}

    bool operator()(const structured_control_flow::ControlFlowNode* lhs, const structured_control_flow::ControlFlowNode* rhs)
        const {
        return std::find(loops_order_->begin(), loops_order_->end(), lhs) <
               std::find(loops_order_->begin(), loops_order_->end(), rhs);
    }
};

class LoopAnalysis : public Analysis {
private:
    std::vector<structured_control_flow::ControlFlowNode*> loops_;
    std::map<structured_control_flow::ControlFlowNode*, structured_control_flow::ControlFlowNode*, DFSLoopComparator>
        loop_tree_;

    void run(structured_control_flow::ControlFlowNode& scope, structured_control_flow::ControlFlowNode* parent_loop);

    std::vector<sdfg::structured_control_flow::ControlFlowNode*> children(
        sdfg::structured_control_flow::ControlFlowNode* node,
        const std::map<
            sdfg::structured_control_flow::ControlFlowNode*,
            sdfg::structured_control_flow::ControlFlowNode*,
            DFSLoopComparator>& tree
    ) const;

    std::list<std::vector<sdfg::structured_control_flow::ControlFlowNode*>> loop_tree_paths(
        sdfg::structured_control_flow::ControlFlowNode* loop,
        const std::map<
            sdfg::structured_control_flow::ControlFlowNode*,
            sdfg::structured_control_flow::ControlFlowNode*,
            DFSLoopComparator>& tree
    ) const;

protected:
    void run(analysis::AnalysisManager& analysis_manager) override;

public:
    LoopAnalysis(StructuredSDFG& sdfg);

    const std::vector<structured_control_flow::ControlFlowNode*> loops() const;

    structured_control_flow::ControlFlowNode* find_loop_by_indvar(const std::string& indvar);

    const std::map<structured_control_flow::ControlFlowNode*, structured_control_flow::ControlFlowNode*, DFSLoopComparator>&
    loop_tree() const;

    structured_control_flow::ControlFlowNode* parent_loop(structured_control_flow::ControlFlowNode* loop) const;

    const std::vector<structured_control_flow::ControlFlowNode*> outermost_loops() const;

    const std::vector<structured_control_flow::ControlFlowNode*> outermost_maps() const;

    std::vector<sdfg::structured_control_flow::ControlFlowNode*> children(sdfg::structured_control_flow::ControlFlowNode*
                                                                              node) const;

    std::list<std::vector<sdfg::structured_control_flow::ControlFlowNode*>>
    loop_tree_paths(sdfg::structured_control_flow::ControlFlowNode* loop) const;

    std::unordered_set<sdfg::structured_control_flow::ControlFlowNode*>
    descendants(sdfg::structured_control_flow::ControlFlowNode* loop) const;

    /**
     * @brief Checks if a loop's update is a strictly monotonic function (positive).
     *
     * @param loop The loop to check.
     * @param assumptions_analysis The assumptions analysis to use.
     * @return True if the loop is monotonic, false otherwise.
     */
    static bool
    is_monotonic(structured_control_flow::StructuredLoop* loop, analysis::AssumptionsAnalysis& assumptions_analysis);

    /**
     * @brief Checks if a loop's update is a contiguous function (positive).
     *
     * @param loop The loop to check.
     * @param assumptions_analysis The assumptions analysis to use.
     * @return True if the loop is contiguous, false otherwise.
     */
    static bool
    is_contiguous(structured_control_flow::StructuredLoop* loop, analysis::AssumptionsAnalysis& assumptions_analysis);

    /**
     * @brief Describes the bound of a loop as a closed-form expression for contiguous loops.
     *
     * Example: i <= N && i < M -> i < min(N + 1, M)
     *
     * @param loop The loop to describe the bound of.
     * @param assumptions_analysis The assumptions analysis to use.
     * @return The bound of the loop as a closed-form expression, otherwise null.
     */
    static symbolic::Expression
    canonical_bound(structured_control_flow::StructuredLoop* loop, analysis::AssumptionsAnalysis& assumptions_analysis);

    /**
     * @brief Describes the stride of a loop's update as a constant.
     *
     * @param loop The loop to describe the stride of.
     * @return The stride of the loop's update as a constant, otherwise null.
     */
    static symbolic::Integer stride(structured_control_flow::StructuredLoop* loop);
};

} // namespace analysis
} // namespace sdfg
