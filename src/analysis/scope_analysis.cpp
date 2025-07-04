#include "sdfg/analysis/scope_analysis.h"

namespace sdfg {
namespace analysis {

ScopeAnalysis::ScopeAnalysis(StructuredSDFG& sdfg) : Analysis(sdfg) {}

void ScopeAnalysis::
    run(structured_control_flow::ControlFlowNode* current, structured_control_flow::ControlFlowNode* parent_scope) {
    if (dynamic_cast<structured_control_flow::Block*>(current)) {
        this->scope_tree_[current] = parent_scope;
    } else if (auto sequence_stmt = dynamic_cast<structured_control_flow::Sequence*>(current)) {
        this->scope_tree_[current] = parent_scope;
        for (size_t i = 0; i < sequence_stmt->size(); i++) {
            this->run(&sequence_stmt->at(i).first, current);
        }
    } else if (auto if_else_stmt = dynamic_cast<structured_control_flow::IfElse*>(current)) {
        this->scope_tree_[current] = parent_scope;
        for (size_t i = 0; i < if_else_stmt->size(); i++) {
            this->run(&if_else_stmt->at(i).first, current);
        }
    } else if (auto while_stmt = dynamic_cast<structured_control_flow::While*>(current)) {
        this->scope_tree_[current] = parent_scope;
        this->run(&while_stmt->root(), current);
    } else if (auto for_stmt = dynamic_cast<structured_control_flow::StructuredLoop*>(current)) {
        this->scope_tree_[current] = parent_scope;
        this->run(&for_stmt->root(), current);
    } else if (dynamic_cast<structured_control_flow::Break*>(current)) {
        this->scope_tree_[current] = parent_scope;
    } else if (dynamic_cast<structured_control_flow::Continue*>(current)) {
        this->scope_tree_[current] = parent_scope;
    } else if (dynamic_cast<structured_control_flow::Return*>(current)) {
        this->scope_tree_[current] = parent_scope;
    } else {
        throw std::runtime_error("Unsupported control flow node type");
    }
}

void ScopeAnalysis::run(AnalysisManager& analysis_manager) {
    this->scope_tree_.clear();
    this->run(&this->sdfg_.root(), nullptr);
}

const std::unordered_map<const structured_control_flow::ControlFlowNode*, structured_control_flow::ControlFlowNode*>&
ScopeAnalysis::scope_tree() const {
    return this->scope_tree_;
}

structured_control_flow::ControlFlowNode* ScopeAnalysis::parent_scope(const structured_control_flow::ControlFlowNode*
                                                                          scope) const {
    if (this->scope_tree_.find(scope) == this->scope_tree_.end()) {
        return nullptr;
    }
    return this->scope_tree_.at(scope);
}

} // namespace analysis
} // namespace sdfg
