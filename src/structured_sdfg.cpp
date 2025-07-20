#include "sdfg/structured_sdfg.h"

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/deepcopy/structured_sdfg_deep_copy.h"
#include "sdfg/element.h"

#include "sdfg/serializer/json_serializer.h"

namespace sdfg {

StructuredSDFG::StructuredSDFG(const std::string& name, FunctionType type) : Function(name, type) {
    this->root_ = std::unique_ptr<
        structured_control_flow::Sequence>(new structured_control_flow::Sequence(this->element_counter_, DebugInfo()));
};

const structured_control_flow::Sequence& StructuredSDFG::root() const { return *this->root_; };

structured_control_flow::Sequence& StructuredSDFG::root() { return *this->root_; };

std::unique_ptr<StructuredSDFG> StructuredSDFG::clone() const {
    serializer::JSONSerializer serializer;
    nlohmann::json j = serializer.serialize(*this);
    return serializer.deserialize(j);
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
        } else if (auto sloop_node = dynamic_cast<const structured_control_flow::StructuredLoop*>(current)) {
            to_visit.insert(&sloop_node->root());
        } else if (auto condition_node = dynamic_cast<const structured_control_flow::IfElse*>(current)) {
            for (size_t i = 0; i < condition_node->size(); i++) {
                to_visit.insert(&condition_node->at(i).first);
            }
        } else if (auto while_node = dynamic_cast<const structured_control_flow::While*>(current)) {
            to_visit.insert(&while_node->root());
        } else if (auto sequence_node = dynamic_cast<const structured_control_flow::Sequence*>(current)) {
            for (size_t i = 0; i < sequence_node->size(); i++) {
                to_visit.insert(&sequence_node->at(i).first);
            }
        } else if (dynamic_cast<const structured_control_flow::Return*>(current)) {
            continue;
        } else if (auto map_node = dynamic_cast<const structured_control_flow::Map*>(current)) {
            to_visit.insert(&map_node->root());
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
        } else if (auto sloop_node = dynamic_cast<const structured_control_flow::StructuredLoop*>(current)) {
            info = DebugInfo::merge(info, sloop_node->debug_info());
            to_visit.insert(&sloop_node->root());
        } else if (auto condition_node = dynamic_cast<const structured_control_flow::IfElse*>(current)) {
            info = DebugInfo::merge(info, condition_node->debug_info());
            for (size_t i = 0; i < condition_node->size(); i++) {
                to_visit.insert(&condition_node->at(i).first);
            }
        } else if (auto while_node = dynamic_cast<const structured_control_flow::While*>(current)) {
            info = DebugInfo::merge(info, while_node->debug_info());
            to_visit.insert(&while_node->root());
        } else if (auto sequence_node = dynamic_cast<const structured_control_flow::Sequence*>(current)) {
            info = DebugInfo::merge(info, sequence_node->debug_info());
            for (size_t i = 0; i < sequence_node->size(); i++) {
                to_visit.insert(&sequence_node->at(i).first);
                info = DebugInfo::merge(info, sequence_node->at(i).second.debug_info());
            }
        } else if (auto return_node = dynamic_cast<const structured_control_flow::Return*>(current)) {
            info = DebugInfo::merge(info, return_node->debug_info());
        } else if (auto map_node = dynamic_cast<const structured_control_flow::Map*>(current)) {
            info = DebugInfo::merge(info, map_node->debug_info());
            to_visit.insert(&map_node->root());
        }
    }
    return info;
};

void StructuredSDFG::validate() const {
    // Call parent validate
    Function::validate();

    // Validate root
    this->root().validate(*this);
};

} // namespace sdfg
