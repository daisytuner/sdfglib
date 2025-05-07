#include "sdfg/serializer/json_serializer.h"

#include <cassert>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/kernel.h"
#include "sdfg/structured_control_flow/return.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/while.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"
#include "symengine/expression.h"
#include "symengine/printers.h"

namespace sdfg {
namespace serializer {

/*
 * * JSONSerializer class
 * * Serialization logic
 */

nlohmann::json JSONSerializer::serialize() {
    nlohmann::json j;

    j["name"] = sdfg_->name();

    j["structures"] = nlohmann::json::array();
    for (const auto& structure : sdfg_->structures()) {
        nlohmann::json structure_json;
        structure_definition_to_json(structure_json, structure);
        j["structures"].push_back(structure_json);
    }
    j["containers"] = nlohmann::json::array();
    for (const auto& container : sdfg_->containers()) {
        nlohmann::json container_json;
        container_json["name"] = container;
        if (sdfg_->is_argument(container)) {
            container_json["argument"] = true;
        } else {
            container_json["argument"] = false;
        }

        if (sdfg_->is_external(container)) {
            container_json["external"] = true;
        } else {
            container_json["external"] = false;
        }

        nlohmann::json container_type_json;
        type_to_json(container_type_json, sdfg_->type(container));
        container_json["type"] = container_type_json;
        j["containers"].push_back(container_json);
    }

    // dump the root node
    nlohmann::json root_json;
    sequence_to_json(root_json, sdfg_->root());
    j["root"] = root_json;

    return j;
}

void JSONSerializer::dataflow_to_json(nlohmann::json& j,
                                      const sdfg::data_flow::DataFlowGraph& dataflow) {
    j["type"] = "dataflow";
    j["nodes"] = nlohmann::json::array();
    j["edges"] = nlohmann::json::array();

    for (auto& node : dataflow.nodes()) {
        nlohmann::json node_json;
        if (auto code_node = dynamic_cast<const sdfg::data_flow::Tasklet*>(&node)) {
            node_json["type"] = "tasklet";
            node_json["code"] = code_node->code();
            node_json["inputs"] = nlohmann::json::array();
            for (auto& input : code_node->inputs()) {
                nlohmann::json input_json;
                nlohmann::json type_json;
                type_to_json(type_json, input.second);
                input_json["type"] = type_json;
                input_json["name"] = input.first;
                node_json["inputs"].push_back(input_json);
            }
            node_json["outputs"] = nlohmann::json::array();
            for (auto& output : code_node->outputs()) {
                nlohmann::json output_json;
                nlohmann::json type_json;
                type_to_json(type_json, output.second);
                output_json["type"] = type_json;
                output_json["name"] = output.first;
                node_json["outputs"].push_back(output_json);
            }
            // node_json["conditional"] = code_node->is_conditional();
            // if (code_node->is_conditional()) {
            //     node_json["condition"] = code_node->condition()->__str__();
            // }
        } else if (auto lib_node = dynamic_cast<const sdfg::data_flow::LibraryNode*>(&node)) {
            node_json["type"] = "library_node";
            node_json["call"] = lib_node->call();
            node_json["side_effect"] = lib_node->has_side_effect();
            node_json["inputs"] = nlohmann::json::array();
            for (auto& input : lib_node->inputs()) {
                nlohmann::json input_json;
                nlohmann::json type_json;
                type_to_json(type_json, input.second);
                input_json["type"] = type_json;
                input_json["name"] = input.first;
                node_json["inputs"].push_back(input_json);
            }
            node_json["outputs"] = nlohmann::json::array();
            for (auto& output : lib_node->outputs()) {
                nlohmann::json output_json;
                nlohmann::json type_json;
                type_to_json(type_json, output.second);
                output_json["type"] = type_json;
                output_json["name"] = output.first;
                node_json["outputs"].push_back(output_json);
            }
        } else if (auto code_node = dynamic_cast<const sdfg::data_flow::AccessNode*>(&node)) {
            node_json["type"] = "access_node";
            node_json["container"] = code_node->data();
        } else {
            throw std::runtime_error("Unknown node type");
        }
        node_json["element_id"] = node.element_id();
        j["nodes"].push_back(node_json);
    }

    for (auto& edge : dataflow.edges()) {
        nlohmann::json edge_json;
        edge_json["source"] = edge.src().element_id();
        edge_json["target"] = edge.dst().element_id();

        edge_json["source_connector"] = edge.src_conn();
        edge_json["target_connector"] = edge.dst_conn();

        // add subset
        edge_json["subset"] = nlohmann::json::array();
        for (auto& subset : edge.subset()) {
            edge_json["subset"].push_back(subset->__str__());
        }

        j["edges"].push_back(edge_json);
    }
}

void JSONSerializer::block_to_json(nlohmann::json& j,
                                   const sdfg::structured_control_flow::Block& block) {
    j["type"] = "block";
    nlohmann::json dataflow_json;
    dataflow_to_json(dataflow_json, block.dataflow());
    j["dataflow"] = dataflow_json;
}

void JSONSerializer::for_node_to_json(nlohmann::json& j,
                                      const sdfg::structured_control_flow::For& for_node) {
    j["type"] = "for";
    j["indvar"] = for_node.indvar()->__str__();
    j["init"] = for_node.init()->__str__();
    j["condition"] = for_node.condition()->__str__();
    j["update"] = for_node.update()->__str__();

    nlohmann::json body_json;
    sequence_to_json(body_json, for_node.root());
    j["children"] = body_json;
}

void JSONSerializer::if_else_to_json(nlohmann::json& j,
                                     const sdfg::structured_control_flow::IfElse& if_else_node) {
    j["type"] = "if_else";
    j["branches"] = nlohmann::json::array();
    for (int i = 0; i < if_else_node.size(); i++) {
        nlohmann::json branch_json;
        branch_json["condition"] = if_else_node.at(i).second->__str__();
        nlohmann::json body_json;
        sequence_to_json(body_json, if_else_node.at(i).first);
        branch_json["children"] = body_json;
        j["branches"].push_back(branch_json);
    }
}

void JSONSerializer::while_node_to_json(nlohmann::json& j,
                                        const sdfg::structured_control_flow::While& while_node) {
    j["type"] = "while";

    nlohmann::json body_json;
    sequence_to_json(body_json, while_node.root());
    j["children"] = body_json;
    j["element_id"] = while_node.element_id();
}

void JSONSerializer::break_node_to_json(nlohmann::json& j,
                                        const sdfg::structured_control_flow::Break& break_node) {
    j["type"] = "break";
    j["target"] = break_node.loop().element_id();
}
void JSONSerializer::continue_node_to_json(
    nlohmann::json& j, const sdfg::structured_control_flow::Continue& continue_node) {
    j["type"] = "continue";
    j["target"] = continue_node.loop().element_id();
}

void JSONSerializer::kernel_to_json(nlohmann::json& j,
                                    const sdfg::structured_control_flow::Kernel& kernel_node) {
    j["type"] = "kernel";
    j["name"] = kernel_node.name();
    j["suffix"] = kernel_node.suffix();

    j["blockDim_x"] = kernel_node.blockDim_x()->__str__();
    j["blockDim_y"] = kernel_node.blockDim_y()->__str__();
    j["blockDim_z"] = kernel_node.blockDim_z()->__str__();
    j["gridDim_x"] = kernel_node.gridDim_x()->__str__();
    j["gridDim_y"] = kernel_node.gridDim_y()->__str__();
    j["gridDim_z"] = kernel_node.gridDim_z()->__str__();

    j["threadIdx_x"] = kernel_node.threadIdx_x()->__str__();
    j["threadIdx_y"] = kernel_node.threadIdx_y()->__str__();
    j["threadIdx_z"] = kernel_node.threadIdx_z()->__str__();
    j["blockIdx_x"] = kernel_node.blockIdx_x()->__str__();
    j["blockIdx_y"] = kernel_node.blockIdx_y()->__str__();
    j["blockIdx_z"] = kernel_node.blockIdx_z()->__str__();

    nlohmann::json body_json;
    sequence_to_json(body_json, kernel_node.root());
    j["children"] = body_json;
}

void JSONSerializer::return_node_to_json(nlohmann::json& j,
                                         const sdfg::structured_control_flow::Return& return_node) {
    j["type"] = "return";
}

void JSONSerializer::sequence_to_json(nlohmann::json& j,
                                      const sdfg::structured_control_flow::Sequence& sequence) {
    j["type"] = "sequence";
    j["children"] = nlohmann::json::array();

    for (size_t i = 0; i < sequence.size(); i++) {
        nlohmann::json child_json;
        auto& child = sequence.at(i).first;
        auto& transition = sequence.at(i).second;

        if (auto block = dynamic_cast<const sdfg::structured_control_flow::Block*>(&child)) {
            block_to_json(child_json, *block);
        } else if (auto for_node =
                       dynamic_cast<const sdfg::structured_control_flow::For*>(&child)) {
            for_node_to_json(child_json, *for_node);
        } else if (auto sequence_node =
                       dynamic_cast<const sdfg::structured_control_flow::Sequence*>(&child)) {
            sequence_to_json(child_json, *sequence_node);
        } else if (auto condition_node =
                       dynamic_cast<const sdfg::structured_control_flow::IfElse*>(&child)) {
            if_else_to_json(child_json, *condition_node);
        } else if (auto while_node =
                       dynamic_cast<const sdfg::structured_control_flow::While*>(&child)) {
            while_node_to_json(child_json, *while_node);
        } else if (auto kernel_node =
                       dynamic_cast<const sdfg::structured_control_flow::Kernel*>(&child)) {
            kernel_to_json(child_json, *kernel_node);
        } else if (auto return_node =
                       dynamic_cast<const sdfg::structured_control_flow::Return*>(&child)) {
            return_node_to_json(child_json, *return_node);
        } else if (auto break_node =
                       dynamic_cast<const sdfg::structured_control_flow::Break*>(&child)) {
            break_node_to_json(child_json, *break_node);
        } else if (auto continue_node =
                       dynamic_cast<const sdfg::structured_control_flow::Continue*>(&child)) {
            continue_node_to_json(child_json, *continue_node);
        } else {
            throw std::runtime_error("Unknown child type");
        }

        // Add transition information
        nlohmann::json transition_json;
        transition_json["type"] = "transition";
        transition_json["assignments"] = nlohmann::json::array();
        for (const auto& assignment : transition.assignments()) {
            nlohmann::json assignment_json;
            assignment_json["symbol"] = assignment.first->__str__();
            assignment_json["expression"] = assignment.second->__str__();
            transition_json["assignments"].push_back(assignment_json);
        }
        j["children"].push_back(child_json);
    }
}

void JSONSerializer::type_to_json(nlohmann::json& j, const sdfg::types::IType& type) {
    if (auto scalar_type = dynamic_cast<const sdfg::types::Scalar*>(&type)) {
        j["type"] = "scalar";
        j["primitive_type"] = scalar_type->primitive_type();
        j["address_space"] = scalar_type->address_space();
        j["initializer"] = scalar_type->initializer();
        j["device_location"] = scalar_type->device_location();
    } else if (auto array_type = dynamic_cast<const sdfg::types::Array*>(&type)) {
        j["type"] = "array";
        nlohmann::json element_type_json;
        type_to_json(element_type_json, array_type->element_type());
        j["element_type"] = element_type_json;
        j["num_elements"] = array_type->num_elements()->__str__();
        j["address_space"] = array_type->address_space();
        j["initializer"] = array_type->initializer();
        j["device_location"] = array_type->device_location();
    } else if (auto pointer_type = dynamic_cast<const sdfg::types::Pointer*>(&type)) {
        j["type"] = "pointer";
        nlohmann::json pointee_type_json;
        type_to_json(pointee_type_json, pointer_type->pointee_type());
        j["pointee_type"] = pointee_type_json;
        j["address_space"] = pointer_type->address_space();
        j["initializer"] = pointer_type->initializer();
        j["device_location"] = pointer_type->device_location();
    } else if (auto structure_type = dynamic_cast<const sdfg::types::Structure*>(&type)) {
        j["type"] = "structure";
        j["name"] = structure_type->name();
        j["address_space"] = structure_type->address_space();
        j["initializer"] = structure_type->initializer();
        j["device_location"] = structure_type->device_location();
    } else {
        throw std::runtime_error("Unknown type");
    }
}

void JSONSerializer::structure_definition_to_json(
    nlohmann::json& j, const sdfg::types::StructureDefinition& definition) {
    j["name"] = definition.name();
    j["members"] = nlohmann::json::array();
    for (size_t i = 0; i < definition.num_members(); i++) {
        nlohmann::json member_json;
        type_to_json(member_json, definition.member_type(sdfg::symbolic::integer(i)));
        j["members"].push_back(member_json);
    }
}

/*
 * * Deserialization logic
 */

std::unique_ptr<sdfg::StructuredSDFG> JSONSerializer::deserialize() {
    std::ifstream file(filename_);
    if (file.is_open()) {
        nlohmann::json j;
        file >> j;
        file.close();

        // Deserialize the JSON data into the StructuredSDFG object
        // This part is not implemented in this example
        // You would need to implement the logic to convert the JSON data back into the
        // StructuredSDFG object
        assert(j.contains("name"));
        sdfg::builder::StructuredSDFGBuilder builder(j["name"]);
        // TODO: implement the deserialization logic for structures, containers, arguments,
        // externals, and root

        return builder.move();

    } else {
        throw std::runtime_error("Could not open file " + filename_);
    }
}

void JSONSerializer::json_to_structure_definition(const nlohmann::json& j,
                                                  sdfg::builder::StructuredSDFGBuilder& builder) {
    assert(j.contains("name"));
    assert(j["name"].is_string());
    assert(j.contains("members"));
    assert(j["members"].is_array());
    auto& definition = builder.add_structure(j["name"]);
    for (const auto& member : j["members"]) {
        nlohmann::json member_json;
        auto member_type = json_to_type(member);
        definition.add_member(*member_type);
    }
}

void JSONSerializer::json_to_containers(const nlohmann::json& j,
                                        sdfg::builder::StructuredSDFGBuilder& builder) {
    assert(j.contains("containers"));
    assert(j["containers"].is_array());
    for (const auto& container : j["containers"]) {
        assert(container.contains("name"));
        assert(container["name"].is_string());
        assert(container.contains("external"));
        assert(container["external"].is_boolean());
        assert(container.contains("argument"));
        assert(container["argument"].is_boolean());
        assert(container.contains("type"));
        assert(container["type"].is_object());
        std::string name = container["name"];
        bool external = container["external"];
        bool argument = container["argument"];
        auto container_type = json_to_type(container["type"]);
        builder.add_container(name, *container_type, argument, external);
    }
}

std::vector<std::pair<std::string, types::Scalar>> JSONSerializer::json_to_arguments(
    const nlohmann::json& j) {
    std::vector<std::pair<std::string, types::Scalar>> arguments;
    for (const auto& argument : j) {
        assert(argument.contains("name"));
        assert(argument["name"].is_string());
        assert(argument.contains("type"));
        assert(argument["type"].is_object());
        std::string name = argument["name"];
        auto type = json_to_type(argument["type"]);
        arguments.emplace_back(name, *dynamic_cast<types::Scalar*>(type.get()));
    }
    return arguments;
}

void JSONSerializer::json_to_dataflow(const nlohmann::json& j,
                                      sdfg::builder::StructuredSDFGBuilder& builder,
                                      sdfg::structured_control_flow::Block& parent) {
    std::unordered_map<long, sdfg::data_flow::DataFlowNode&> nodes_map;

    assert(j.contains("nodes"));
    assert(j["nodes"].is_array());
    for (const auto& node : j["nodes"]) {
        assert(node.contains("type"));
        assert(node["type"].is_string());
        assert(node.contains("element_id"));
        assert(node["element_id"].is_number_integer());
        std::string type = node["type"];
        if (type == "tasklet") {
            assert(node.contains("code"));
            assert(node["code"].is_number_integer());
            assert(node.contains("inputs"));
            assert(node["inputs"].is_array());
            assert(node.contains("outputs"));
            assert(node["outputs"].is_array());
            auto outputs = json_to_arguments(node["outputs"]);
            assert(outputs.size() == 1);
            auto inputs = json_to_arguments(node["inputs"]);
            auto& tasklet = builder.add_tasklet(parent, node["code"], outputs.at(0), inputs);

            nodes_map.insert({node["element_id"], tasklet});
        } else if (type == "library_node") {
            assert(node.contains("call"));
            assert(node.contains("inputs"));
            assert(node["inputs"].is_array());
            assert(node.contains("outputs"));
            assert(node["outputs"].is_array());
            auto outputs = json_to_arguments(node["outputs"]);
            auto inputs = json_to_arguments(node["inputs"]);
            auto& lib_node = builder.add_library_node(parent, node["call"], outputs, inputs);

            nodes_map.insert({node["element_id"], lib_node});
        } else if (type == "access_node") {
            assert(node.contains("container"));
            auto& access_node = builder.add_access(parent, node["container"]);

            nodes_map.insert({node["element_id"], access_node});
        } else {
            throw std::runtime_error("Unknown node type");
        }
    }

    assert(j.contains("edges"));
    assert(j["edges"].is_array());
    for (const auto& edge : j["edges"]) {
        assert(edge.contains("source"));
        assert(edge["source"].is_number_integer());
        assert(edge.contains("target"));
        assert(edge["target"].is_number_integer());
        assert(edge.contains("source_connector"));
        assert(edge["source_connector"].is_string());
        assert(edge.contains("target_connector"));
        assert(edge["target_connector"].is_string());
        assert(edge.contains("subset"));
        assert(edge["subset"].is_array());

        assert(nodes_map.contains(edge["source"]));
        assert(nodes_map.contains(edge["target"]));
        auto& source = nodes_map.at(edge["source"]);
        auto& target = nodes_map.at(edge["target"]);

        assert(edge.contains("subset"));
        assert(edge["subset"].is_array());
        std::vector<sdfg::symbolic::Expression> subset;
        for (const auto& subset_str : edge["subset"]) {
            assert(subset_str.is_string());
            subset.push_back(SymEngine::Expression(subset_str));
        }
        builder.add_memlet(parent, source, edge["source_connector"], target,
                           edge["target_connector"], subset);
    }
}

void JSONSerializer::json_to_sequence(const nlohmann::json& j,
                                      sdfg::builder::StructuredSDFGBuilder& builder) {}

void JSONSerializer::json_to_block(const nlohmann::json& j,
                                   sdfg::builder::StructuredSDFGBuilder& builder,
                                   sdfg::structured_control_flow::Sequence& parent) {}

void JSONSerializer::json_to_for_node(const nlohmann::json& j,
                                      sdfg::builder::StructuredSDFGBuilder& builder,
                                      sdfg::structured_control_flow::Sequence& parent) {}

void JSONSerializer::json_to_if_else_node(const nlohmann::json& j,
                                          sdfg::builder::StructuredSDFGBuilder& builder,
                                          sdfg::structured_control_flow::Sequence& parent) {}

void JSONSerializer::json_to_while_node(const nlohmann::json& j,
                                        sdfg::builder::StructuredSDFGBuilder& builder,
                                        sdfg::structured_control_flow::Sequence& parent) {}

void JSONSerializer::json_to_break_node(const nlohmann::json& j,
                                        sdfg::builder::StructuredSDFGBuilder& builder,
                                        sdfg::structured_control_flow::Sequence& parent) {}

void JSONSerializer::json_to_continue_node(const nlohmann::json& j,
                                           sdfg::builder::StructuredSDFGBuilder& builder,
                                           sdfg::structured_control_flow::Sequence& parent) {}

void JSONSerializer::json_to_kernel_node(const nlohmann::json& j,
                                         sdfg::builder::StructuredSDFGBuilder& builder,
                                         sdfg::structured_control_flow::Sequence& parent) {}

void JSONSerializer::json_to_return_node(const nlohmann::json& j,
                                         sdfg::builder::StructuredSDFGBuilder& builder,
                                         sdfg::structured_control_flow::Sequence& parent) {}

void JSONSerializer::json_to_transition(const nlohmann::json& j,
                                        sdfg::builder::StructuredSDFGBuilder& builder,
                                        sdfg::structured_control_flow::Sequence& parent) {}

std::unique_ptr<sdfg::types::IType> JSONSerializer::json_to_type(const nlohmann::json& j) {
    if (j.contains("type")) {
        if (j["type"] == "scalar") {
            // Deserialize scalar type
            assert(j.contains("primitive_type"));
            sdfg::types::PrimitiveType primitive_type = j["primitive_type"];
            assert(j.contains("device_location"));
            sdfg::types::DeviceLocation device_location = j["device_location"];
            assert(j.contains("address_space"));
            uint address_space = j["address_space"];
            assert(j.contains("initializer"));
            std::string initializer = j["initializer"];
            return std::make_unique<sdfg::types::Scalar>(primitive_type, device_location,
                                                         address_space, initializer);
        } else if (j["type"] == "array") {
            // Deserialize array type
            assert(j.contains("element_type"));
            std::unique_ptr<sdfg::types::IType> member_type = json_to_type(j["element_type"]);
            assert(j.contains("num_elements"));
            std::string num_elements_str = j["num_elements"];
            // Convert num_elements_str to symbolic::Expression
            sdfg::symbolic::Expression num_elements = SymEngine::Expression(num_elements_str);
            assert(j.contains("address_space"));
            uint address_space = j["address_space"];
            assert(j.contains("initializer"));
            std::string initializer = j["initializer"];
            assert(j.contains("device_location"));
            sdfg::types::DeviceLocation device_location = j["device_location"];
            return std::make_unique<sdfg::types::Array>(*member_type, num_elements, device_location,
                                                        address_space, initializer);
        } else if (j["type"] == "pointer") {
            // Deserialize pointer type
            assert(j.contains("pointee_type"));
            std::unique_ptr<sdfg::types::IType> pointee_type = json_to_type(j["pointee_type"]);
            assert(j.contains("address_space"));
            uint address_space = j["address_space"];
            assert(j.contains("initializer"));
            std::string initializer = j["initializer"];
            assert(j.contains("device_location"));
            sdfg::types::DeviceLocation device_location = j["device_location"];
            return std::make_unique<sdfg::types::Pointer>(*pointee_type, device_location,
                                                          address_space, initializer);
        } else if (j["type"] == "structure") {
            // Deserialize structure type
            assert(j.contains("name"));
            std::string name = j["name"];
            assert(j.contains("address_space"));
            uint address_space = j["address_space"];
            assert(j.contains("initializer"));
            std::string initializer = j["initializer"];
            assert(j.contains("device_location"));
            sdfg::types::DeviceLocation device_location = j["device_location"];
            return std::make_unique<sdfg::types::Structure>(name, device_location, address_space,
                                                            initializer);
        } else {
            throw std::runtime_error("Unknown type");
        }
    } else {
        throw std::runtime_error("Type not found");
    }
}

}  // namespace serializer
}  // namespace sdfg
