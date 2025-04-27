#include "sdfg/serializer/json_serializer.h"

#include <cassert>

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

/*
 * * JSONSerializer class
 * * Serialization logic
 */

void JSONSerializer::serialize() {
    nlohmann::json j;

    j["name"] = sdfg_.name();

    j["structures"] = nlohmann::json::array();
    for (const auto& structure : sdfg_.structures()) {
        nlohmann::json structure_json;
        structure_definition_to_json(structure_json, structure);
        j["structures"].push_back(structure_json);
    }
    j["containers"] = nlohmann::json::array();
    for (const auto& container : sdfg_.containers()) {
        nlohmann::json container_type_json;
        type_to_json(container_type_json, sdfg_.type(container));
        j[container].push_back(container_type_json);
    }
    j["arguments"] = nlohmann::json::array();
    for (const auto& arg : sdfg_.arguments()) {
        j["arguments"].push_back(arg);
    }
    j["externals"] = nlohmann::json::array();
    for (const auto& ext : sdfg_.externals()) {
        j["externals"].push_back(ext);
    }

    // dump the root node
    nlohmann::json root_json;
    sequence_to_json(root_json, sdfg_.root());
    j["root"] = root_json;

    // dump to file
    std::ofstream file(filename_);
    if (file.is_open()) {
        file << j.dump(4);
        file.close();
    } else {
        throw std::runtime_error("Could not open file " + filename_);
    }
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
                node_json["inputs"].push_back(input.first);
            }
            node_json["outputs"] = nlohmann::json::array();
            for (auto& output : code_node->outputs()) {
                node_json["outputs"].push_back(output.first);
            }
            node_json["conditional"] = code_node->is_conditional();
            if (code_node->is_conditional()) {
                node_json["condition"] = code_node->condition()->dumps();
            }
        } else if (auto lib_node = dynamic_cast<const sdfg::data_flow::LibraryNode*>(&node)) {
            node_json["type"] = "library_node";
            node_json["call"] = lib_node->call();
            node_json["side_effect"] = lib_node->has_side_effect();
            node_json["inputs"] = nlohmann::json::array();
            for (auto& input : lib_node->inputs()) {
                node_json["inputs"].push_back(input.first);
            }
            node_json["outputs"] = nlohmann::json::array();
            for (auto& output : lib_node->outputs()) {
                node_json["outputs"].push_back(output.first);
            }
        } else if (auto code_node = dynamic_cast<const sdfg::data_flow::AccessNode*>(&node)) {
            node_json["type"] = "access_node";
            node_json["name"] = code_node->name();
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
            edge_json["subset"].push_back(subset->dumps());
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
    j["indvar"] = for_node.indvar()->dumps();
    j["init"] = for_node.init()->dumps();
    j["condition"] = for_node.condition()->dumps();
    j["update"] = for_node.update()->dumps();

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
        branch_json["condition"] = if_else_node.at(i).second->dumps();
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
}

void JSONSerializer::break_node_to_json(nlohmann::json& j,
                                        const sdfg::structured_control_flow::Break& break_node) {
    j["type"] = "break";
}
void JSONSerializer::continue_node_to_json(
    nlohmann::json& j, const sdfg::structured_control_flow::Continue& continue_node) {
    j["type"] = "continue";
}

void JSONSerializer::kernel_to_json(nlohmann::json& j,
                                    const sdfg::structured_control_flow::Kernel& kernel_node) {
    j["type"] = "kernel";
    j["name"] = kernel_node.name();
    j["inputs"] = nlohmann::json::array();
    j["suffix"] = kernel_node.suffix();

    j["blockDim_x"] = kernel_node.blockDim_x()->dumps();
    j["blockDim_y"] = kernel_node.blockDim_y()->dumps();
    j["blockDim_z"] = kernel_node.blockDim_z()->dumps();
    j["gridDim_x"] = kernel_node.gridDim_x()->dumps();
    j["gridDim_y"] = kernel_node.gridDim_y()->dumps();
    j["gridDim_z"] = kernel_node.gridDim_z()->dumps();

    j["threadIdx_x"] = kernel_node.threadIdx_x()->dumps();
    j["threadIdx_y"] = kernel_node.threadIdx_y()->dumps();
    j["threadIdx_z"] = kernel_node.threadIdx_z()->dumps();
    j["blockIdx_x"] = kernel_node.blockIdx_x()->dumps();
    j["blockIdx_y"] = kernel_node.blockIdx_y()->dumps();
    j["blockIdx_z"] = kernel_node.blockIdx_z()->dumps();
}

void JSONSerializer::return_node_to_json(nlohmann::json& j,
                                         const sdfg::structured_control_flow::Return& return_node) {
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
        } else {
            throw std::runtime_error("Unknown child type");
        }

        // Add transition information
        nlohmann::json transition_json;
        transition_json["type"] = "transition";
        transition_json["assignments"] = nlohmann::json::array();
        for (const auto& assignment : transition.assignments()) {
            nlohmann::json assignment_json;
            assignment_json["symbol"] = assignment.first->dumps();
            assignment_json["expression"] = assignment.second->dumps();
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
        j["num_elements"] = array_type->num_elements()->dumps();
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

void JSONSerializer::deserialize() {
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

    } else {
        throw std::runtime_error("Could not open file " + filename_);
    }
}
