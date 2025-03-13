#pragma once

#include <algorithm>

#include "sdfg/passes/pass.h"

namespace sdfg {
namespace passes {

class StrideMinimization : public Pass {
   private:
    std::pair<bool, std::vector<std::string>> can_be_applied(
        Schedule& schedule, std::vector<structured_control_flow::ControlFlowNode*>& nested_loops);

    void apply(Schedule& schedule,
               std::vector<structured_control_flow::ControlFlowNode*>& nested_loops,
               std::vector<std::string> target_permutation);

    std::vector<structured_control_flow::ControlFlowNode*> children(
        structured_control_flow::ControlFlowNode* node,
        const std::unordered_map<structured_control_flow::ControlFlowNode*,
                                 structured_control_flow::ControlFlowNode*>& tree) const;

    std::list<std::vector<structured_control_flow::ControlFlowNode*>> loop_tree_paths(
        structured_control_flow::ControlFlowNode* loop,
        const std::unordered_map<structured_control_flow::ControlFlowNode*,
                                 structured_control_flow::ControlFlowNode*>& tree) const;

   public:
    StrideMinimization();

    std::string name() override;

    virtual bool run_pass(Schedule& schedule) override;

    static bool is_admissible(std::vector<std::string>& current, std::vector<std::string>& target,
                              std::unordered_set<std::string>& allowed_swaps);

    static std::unordered_set<std::string> allowed_swaps(
        Schedule& schedule, std::vector<structured_control_flow::ControlFlowNode*>& nested_loops);
};

}  // namespace passes
}  // namespace sdfg
