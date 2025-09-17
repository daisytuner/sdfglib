#include "sdfg/structured_sdfg.h"

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/deepcopy/structured_sdfg_deep_copy.h"
#include "sdfg/element.h"

#include "sdfg/serializer/json_serializer.h"

namespace sdfg {

StructuredSDFG::StructuredSDFG(const std::string& name, FunctionType type, const types::IType& return_type)
    : Function(name, type, return_type) {
    this->root_ = std::unique_ptr<
        structured_control_flow::Sequence>(new structured_control_flow::Sequence(this->element_counter_, DebugInfo()));
};

StructuredSDFG::StructuredSDFG(const std::string& name, FunctionType type)
    : StructuredSDFG(name, type, types::Scalar(types::PrimitiveType::Void)) {}

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

void StructuredSDFG::validate() const {
    // Call parent validate
    Function::validate();

    // Validate root
    this->root().validate(*this);
};

} // namespace sdfg
