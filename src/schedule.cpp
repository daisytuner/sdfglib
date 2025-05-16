#include "sdfg/schedule.h"

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/deepcopy/structured_sdfg_deep_copy.h"

namespace sdfg {

void Schedule::loop_tree(
    structured_control_flow::ControlFlowNode& root,
    structured_control_flow::ControlFlowNode* parent,
    std::unordered_map<structured_control_flow::ControlFlowNode*,
                       structured_control_flow::ControlFlowNode*>& tree) const {
    std::list<structured_control_flow::ControlFlowNode*> queue = {&root};
    while (!queue.empty()) {
        auto current = queue.front();
        queue.pop_front();
        if (auto for_loop = dynamic_cast<structured_control_flow::For*>(current)) {
            tree[for_loop] = parent;
        } else if (auto while_loop = dynamic_cast<structured_control_flow::While*>(current)) {
            tree[while_loop] = parent;
        }

        if (auto sequence_stmt = dynamic_cast<structured_control_flow::Sequence*>(current)) {
            for (size_t i = 0; i < sequence_stmt->size(); i++) {
                queue.push_back(&sequence_stmt->at(i).first);
            }
        } else if (auto if_else_stmt = dynamic_cast<structured_control_flow::IfElse*>(current)) {
            for (size_t i = 0; i < if_else_stmt->size(); i++) {
                queue.push_back(&if_else_stmt->at(i).first);
            }
        } else if (auto while_stmt = dynamic_cast<structured_control_flow::While*>(current)) {
            this->loop_tree(while_stmt->root(), while_stmt, tree);
        } else if (auto for_stmt = dynamic_cast<structured_control_flow::For*>(current)) {
            this->loop_tree(for_stmt->root(), for_stmt, tree);
        } else if (auto kern_stmt = dynamic_cast<structured_control_flow::Kernel*>(current)) {
            queue.push_back(&kern_stmt->root());
        }
    }
};

Schedule::Schedule(std::unique_ptr<StructuredSDFG>& sdfg)
    : assumptions_(),
      builder_(sdfg),
      analysis_manager_(builder_.subject(), assumptions_){

      };

Schedule::Schedule(std::unique_ptr<StructuredSDFG>& sdfg, const symbolic::Assumptions& assumptions)
    : assumptions_(assumptions),
      builder_(sdfg),
      analysis_manager_(builder_.subject(), assumptions_){

      };

symbolic::Assumptions& Schedule::assumptions() { return this->assumptions_; };

const symbolic::Assumptions& Schedule::assumptions() const { return this->assumptions_; };

builder::StructuredSDFGBuilder& Schedule::builder() { return this->builder_; };

const StructuredSDFG& Schedule::sdfg() const { return this->builder_.subject(); };

analysis::AnalysisManager& Schedule::analysis_manager() { return this->analysis_manager_; };

LoopSchedule Schedule::loop_schedule(
    const structured_control_flow::ControlFlowNode* loop) const {
    if (this->loop_schedules_.find(loop) == this->loop_schedules_.end()) {
        return LoopSchedule::SEQUENTIAL;
    }
    return this->loop_schedules_.at(loop);
};

void Schedule::loop_schedule(const structured_control_flow::ControlFlowNode* loop,
                             const LoopSchedule schedule) {
    if (schedule == LoopSchedule::SEQUENTIAL) {
        if (this->loop_schedules_.find(loop) == this->loop_schedules_.end()) {
            return;
        }
        this->loop_schedules_.erase(loop);
        return;
    }
    this->loop_schedules_.insert_or_assign(loop, schedule);
};

std::unordered_map<structured_control_flow::ControlFlowNode*,
                   structured_control_flow::ControlFlowNode*>
Schedule::loop_tree() const {
    std::unordered_map<structured_control_flow::ControlFlowNode*,
                       structured_control_flow::ControlFlowNode*>
        loop_tree_;
    this->loop_tree(this->builder_.subject().root(), nullptr, loop_tree_);
    return loop_tree_;
};

/***** Allocation Management *****/

const structured_control_flow::ControlFlowNode* Schedule::allocation_lifetime(
    const std::string& container) const {
    if (this->allocation_lifetimes_.find(container) == this->allocation_lifetimes_.end()) {
        return &this->sdfg().root();
    }
    return this->allocation_lifetimes_.at(container);
};

void Schedule::allocation_lifetime(const std::string& container,
                                   const structured_control_flow::ControlFlowNode* node) {
    this->allocation_lifetimes_.insert_or_assign(container, node);
};

AllocationType Schedule::allocation_type(const std::string& container) const {
    if (this->allocation_types_.find(container) == this->allocation_types_.end()) {
        return AllocationType::DECLARE;
    }
    return this->allocation_types_.at(container);
};

void Schedule::allocation_type(const std::string& container, AllocationType allocation_type) {
    if (allocation_type == AllocationType::DECLARE) {
        if (this->allocation_types_.find(container) != this->allocation_types_.end()) {
            this->allocation_types_.erase(container);
        }
        return;
    }
    this->allocation_types_.insert_or_assign(container, allocation_type);
};

std::unordered_set<std::string> Schedule::node_allocations(
    const structured_control_flow::ControlFlowNode* node) const {
    std::unordered_set<std::string> allocated;
    if (node == &this->sdfg().root()) {
        for (auto& container : this->sdfg().containers()) {
            if (!this->sdfg().is_transient(container)) {
                continue;
            }
            if (this->allocation_lifetimes_.find(container) == this->allocation_lifetimes_.end()) {
                allocated.insert(container);
            }
        }
    } else {
        for (auto& [container, lifetime] : this->allocation_lifetimes_) {
            if (lifetime == node) {
                allocated.insert(container);
            }
        }
    }
    return allocated;
};

std::unordered_set<std::string> Schedule::allocations(
    const structured_control_flow::ControlFlowNode* node) const {
    std::unordered_set<std::string> allocated;

    std::list<const structured_control_flow::ControlFlowNode*> queue = {node};
    while (!queue.empty()) {
        auto current = queue.front();
        queue.pop_front();

        auto node_allocations = this->node_allocations(current);
        allocated.insert(node_allocations.begin(), node_allocations.end());

        if (auto sequence_stmt = dynamic_cast<const structured_control_flow::Sequence*>(current)) {
            for (size_t i = 0; i < sequence_stmt->size(); i++) {
                queue.push_front(&sequence_stmt->at(i).first);
            }
        } else if (auto if_else_stmt =
                       dynamic_cast<const structured_control_flow::IfElse*>(current)) {
            for (size_t i = 0; i < if_else_stmt->size(); i++) {
                queue.push_front(&if_else_stmt->at(i).first);
            }
        } else if (auto while_stmt = dynamic_cast<const structured_control_flow::While*>(current)) {
            queue.push_front(&while_stmt->root());
        } else if (auto for_stmt = dynamic_cast<const structured_control_flow::For*>(current)) {
            queue.push_front(&for_stmt->root());
        }
    }

    return allocated;
};

}  // namespace sdfg
