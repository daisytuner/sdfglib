#include "sdfg/analysis/scope_tree_analysis.h"

namespace sdfg {
namespace analysis {

ScopeTreeAnalysis::ScopeTreeAnalysis(StructuredSDFG& sdfg) : Analysis(sdfg) {}

void ScopeTreeAnalysis::run(structured_control_flow::ControlFlowNode* current,
                            structured_control_flow::ControlFlowNode* parent_scope) {
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
    } else if (auto for_stmt = dynamic_cast<structured_control_flow::For*>(current)) {
        this->scope_tree_[current] = parent_scope;
        this->run(&for_stmt->root(), current);
    } else if (dynamic_cast<structured_control_flow::Kernel*>(current)) {
        this->scope_tree_[current] = parent_scope;
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

void ScopeTreeAnalysis::run(AnalysisManager& analysis_manager) {
    this->scope_tree_.clear();

    this->scope_tree_[&this->sdfg_.root()] = nullptr;
    this->run(&this->sdfg_.root(), nullptr);
}

const std::unordered_map<structured_control_flow::ControlFlowNode*,
                         structured_control_flow::ControlFlowNode*>&
ScopeTreeAnalysis::scope_tree() const {
    return this->scope_tree_;
}

structured_control_flow::ControlFlowNode* ScopeTreeAnalysis::parent_scope(
    structured_control_flow::ControlFlowNode* scope) const {
    return this->scope_tree_.at(scope);
}

}  // namespace analysis
}  // namespace sdfg