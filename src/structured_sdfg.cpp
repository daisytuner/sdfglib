#include "sdfg/structured_sdfg.h"

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/deepcopy/structured_sdfg_deep_copy.h"
#include "sdfg/element.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"

namespace sdfg {

StructuredSDFG::StructuredSDFG(const std::string& name) : Function(name) {
    size_t element_id = 0;
    this->root_ = std::unique_ptr<structured_control_flow::Sequence>(
        new structured_control_flow::Sequence(element_id, DebugInfo()));
};

const structured_control_flow::Sequence& StructuredSDFG::root() const { return *this->root_; };

structured_control_flow::Sequence& StructuredSDFG::root() { return *this->root_; };

std::unique_ptr<StructuredSDFG> StructuredSDFG::clone() const {
    builder::StructuredSDFGBuilder builder(this->name_);
    auto& new_sdfg = builder.subject();

    for (auto& structure : this->structures_) {
        new_sdfg.structures_.insert({structure.first, structure.second->clone()});
    }

    for (auto& container : this->containers_) {
        new_sdfg.containers_.insert({container.first, container.second->clone()});
    }

    for (auto& arg : this->arguments_) {
        new_sdfg.arguments_.push_back(arg);
    }

    for (auto& ext : this->externals_) {
        new_sdfg.externals_.push_back(ext);
    }

    for (auto& entry : this->metadata_) {
        new_sdfg.metadata_[entry.first] = entry.second;
    }

    for (auto& assumption : this->assumptions_) {
        new_sdfg.assumptions_.insert({assumption.first, assumption.second});
    }

    deepcopy::StructuredSDFGDeepCopy copier(builder, new_sdfg.root(), *this->root_);
    auto mapping = copier.insert();

    return builder.move();
};

size_t StructuredSDFG::num_nodes() const {
    size_t count = 0;
    std::set<const ControlFlowNode*> to_visit = {&this->root()};
    while (!to_visit.empty()) {
        auto current = *to_visit.begin();
        to_visit.erase(to_visit.begin());
        // if instance of block, add children to to_visit
        if (auto block = dynamic_cast<const structured_control_flow::Block*>(current)) {
            count += block->dataflow().nodes().size();
        } else if (auto sloop_node =
                       dynamic_cast<const structured_control_flow::StructuredLoop*>(current)) {
            to_visit.insert(&sloop_node->root());
        } else if (auto condition_node =
                       dynamic_cast<const structured_control_flow::IfElse*>(current)) {
            for (size_t i = 0; i < condition_node->size(); i++) {
                to_visit.insert(&condition_node->at(i).first);
            }
        } else if (auto while_node = dynamic_cast<const structured_control_flow::While*>(current)) {
            to_visit.insert(&while_node->root());
        } else if (auto sequence_node =
                       dynamic_cast<const structured_control_flow::Sequence*>(current)) {
            for (size_t i = 0; i < sequence_node->size(); i++) {
                to_visit.insert(&sequence_node->at(i).first);
            }
        } else if (auto kernel_node =
                       dynamic_cast<const structured_control_flow::Kernel*>(current)) {
            to_visit.insert(&kernel_node->root());
        } else if (dynamic_cast<const structured_control_flow::Return*>(current)) {
            continue;
        }
    }
    return count;
};

const DebugInfo StructuredSDFG::debug_info() const {
    DebugInfo info;
    std::set<const ControlFlowNode*> to_visit = {&this->root()};
    while (!to_visit.empty()) {
        auto current = *to_visit.begin();
        to_visit.erase(to_visit.begin());
        info = DebugInfo::merge(info, current->debug_info());

        // if instance of block, add children to to_visit
        if (auto block = dynamic_cast<const structured_control_flow::Block*>(current)) {
            for (auto& node : block->dataflow().nodes()) {
                info = DebugInfo::merge(info, node.debug_info());
            }
            for (auto& edge : block->dataflow().edges()) {
                info = DebugInfo::merge(info, edge.debug_info());
            }
        } else if (auto sloop_node =
                       dynamic_cast<const structured_control_flow::StructuredLoop*>(current)) {
            info = DebugInfo::merge(info, sloop_node->debug_info());
            to_visit.insert(&sloop_node->root());
        } else if (auto condition_node =
                       dynamic_cast<const structured_control_flow::IfElse*>(current)) {
            info = DebugInfo::merge(info, condition_node->debug_info());
            for (size_t i = 0; i < condition_node->size(); i++) {
                to_visit.insert(&condition_node->at(i).first);
            }
        } else if (auto while_node = dynamic_cast<const structured_control_flow::While*>(current)) {
            info = DebugInfo::merge(info, while_node->debug_info());
            to_visit.insert(&while_node->root());
        } else if (auto sequence_node =
                       dynamic_cast<const structured_control_flow::Sequence*>(current)) {
            info = DebugInfo::merge(info, sequence_node->debug_info());
            for (size_t i = 0; i < sequence_node->size(); i++) {
                to_visit.insert(&sequence_node->at(i).first);
                info = DebugInfo::merge(info, sequence_node->at(i).second.debug_info());
            }
        } else if (auto kernel_node =
                       dynamic_cast<const structured_control_flow::Kernel*>(current)) {
            info = DebugInfo::merge(info, kernel_node->debug_info());
            to_visit.insert(&kernel_node->root());
        } else if (auto return_node =
                       dynamic_cast<const structured_control_flow::Return*>(current)) {
            info = DebugInfo::merge(info, return_node->debug_info());
        }
    }
    return info;
};

}  // namespace sdfg
