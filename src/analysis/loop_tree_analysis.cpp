#include "sdfg/analysis/loop_tree_analysis.h"

namespace sdfg {
namespace analysis {

LoopTreeAnalysis::LoopTreeAnalysis(StructuredSDFG& sdfg) : Analysis(sdfg) {}

void LoopTreeAnalysis::run(structured_control_flow::ControlFlowNode& scope,
                           structured_control_flow::ControlFlowNode* parent_loop) {
    std::list<structured_control_flow::ControlFlowNode*> queue = {&scope};
    while (!queue.empty()) {
        auto current = queue.front();
        queue.pop_front();

        // Loop detected
        if (auto while_stmt = dynamic_cast<structured_control_flow::While*>(current)) {
            this->loop_tree_[while_stmt] = parent_loop;
        } else if (auto for_stmt = dynamic_cast<structured_control_flow::For*>(current)) {
            this->loop_tree_[for_stmt] = parent_loop;
        }

        if (dynamic_cast<structured_control_flow::Block*>(current)) {
            continue;
        } else if (auto sequence_stmt = dynamic_cast<structured_control_flow::Sequence*>(current)) {
            for (size_t i = 0; i < sequence_stmt->size(); i++) {
                queue.push_back(&sequence_stmt->at(i).first);
            }
        } else if (auto if_else_stmt = dynamic_cast<structured_control_flow::IfElse*>(current)) {
            for (size_t i = 0; i < if_else_stmt->size(); i++) {
                queue.push_back(&if_else_stmt->at(i).first);
            }
        } else if (auto while_stmt = dynamic_cast<structured_control_flow::While*>(current)) {
            this->run(while_stmt->root(), while_stmt);
        } else if (auto for_stmt = dynamic_cast<structured_control_flow::For*>(current)) {
            this->run(for_stmt->root(), for_stmt);
        } else if (dynamic_cast<structured_control_flow::Break*>(current)) {
            continue;
        } else if (dynamic_cast<structured_control_flow::Continue*>(current)) {
            continue;
        } else if (dynamic_cast<structured_control_flow::Return*>(current)) {
            continue;
        } else {
            throw std::runtime_error("Unsupported control flow node type");
        }
    }
}

void LoopTreeAnalysis::run(AnalysisManager& analysis_manager) {
    this->loop_tree_.clear();
    this->run(this->sdfg_.root(), nullptr);
}

const std::unordered_map<structured_control_flow::ControlFlowNode*,
                         structured_control_flow::ControlFlowNode*>&
LoopTreeAnalysis::loop_tree() const {
    return this->loop_tree_;
}

structured_control_flow::ControlFlowNode* LoopTreeAnalysis::parent_loop(
    structured_control_flow::ControlFlowNode* loop) const {
    return this->loop_tree_.at(loop);
}

const std::vector<structured_control_flow::ControlFlowNode*> LoopTreeAnalysis::outermost_loops()
    const {
    std::vector<structured_control_flow::ControlFlowNode*> outermost_loops_;
    for (const auto& [loop, parent] : this->loop_tree_) {
        if (parent == nullptr) {
            outermost_loops_.push_back(loop);
        }
    }
    return outermost_loops_;
}

}  // namespace analysis
}  // namespace sdfg