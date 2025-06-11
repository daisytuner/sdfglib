#include "sdfg/analysis/loop_analysis.h"

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/functions.h"

namespace sdfg {
namespace analysis {

LoopAnalysis::LoopAnalysis(StructuredSDFG& sdfg) : Analysis(sdfg) {}

void LoopAnalysis::run(structured_control_flow::ControlFlowNode& scope,
                       structured_control_flow::ControlFlowNode* parent_loop) {
    std::list<structured_control_flow::ControlFlowNode*> queue = {&scope};
    while (!queue.empty()) {
        auto current = queue.front();
        queue.pop_front();

        // Loop detected
        if (auto while_stmt = dynamic_cast<structured_control_flow::While*>(current)) {
            this->loops_.insert(while_stmt);
            this->loop_tree_[while_stmt] = parent_loop;
        } else if (auto loop_stmt =
                       dynamic_cast<structured_control_flow::StructuredLoop*>(current)) {
            this->loops_.insert(loop_stmt);
            this->loop_tree_[loop_stmt] = parent_loop;
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
        } else if (auto for_stmt =
                       dynamic_cast<structured_control_flow::StructuredLoop*>(current)) {
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

void LoopAnalysis::run(AnalysisManager& analysis_manager) {
    this->loops_.clear();
    this->loop_tree_.clear();
    this->run(this->sdfg_.root(), nullptr);
}

const std::unordered_set<structured_control_flow::ControlFlowNode*> LoopAnalysis::loops() const {
    return this->loops_;
}

bool LoopAnalysis::is_monotonic(const structured_control_flow::StructuredLoop* loop) const {
    AnalysisManager manager(this->sdfg_);
    auto& assums_analysis = manager.get<AssumptionsAnalysis>();
    auto assums = assums_analysis.get(*loop);

    return symbolic::is_monotonic(loop->update(), loop->indvar(), assums);
}

bool LoopAnalysis::is_contiguous(const structured_control_flow::StructuredLoop* loop) const {
    AnalysisManager manager(this->sdfg_);
    auto& assums_analysis = manager.get<AssumptionsAnalysis>();
    auto assums = assums_analysis.get(*loop);

    return symbolic::is_contiguous(loop->update(), loop->indvar(), assums);
}

const std::unordered_map<structured_control_flow::ControlFlowNode*,
                         structured_control_flow::ControlFlowNode*>&
LoopAnalysis::loop_tree() const {
    return this->loop_tree_;
}

structured_control_flow::ControlFlowNode* LoopAnalysis::parent_loop(
    structured_control_flow::ControlFlowNode* loop) const {
    return this->loop_tree_.at(loop);
}

const std::vector<structured_control_flow::ControlFlowNode*> LoopAnalysis::outermost_loops() const {
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