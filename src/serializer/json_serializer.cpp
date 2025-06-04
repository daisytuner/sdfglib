#include "sdfg/serializer/json_serializer.h"

#include <cassert>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/element.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/kernel.h"
#include "sdfg/structured_control_flow/return.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/while.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/function.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"
#include "symengine/expression.h"
#include "symengine/logic.h"
#include "symengine/symengine_rcp.h"

namespace sdfg {
namespace serializer {

/*
 * * JSONSerializer class
 * * Serialization logic
 */

nlohmann::json JSONSerializer::serialize(std::unique_ptr<sdfg::StructuredSDFG>& sdfg) {
    nlohmann::json j;

    j["name"] = sdfg->name();

    j["structures"] = nlohmann::json::array();
    for (const auto& structure_name : sdfg->structures()) {
        const auto& structure = sdfg->structure(structure_name);
        nlohmann::json structure_json;
        structure_definition_to_json(structure_json, structure);
        j["structures"].push_back(structure_json);
    }

    j["containers"] = nlohmann::json::object();
    for (const auto& container : sdfg->containers()) {
        nlohmann::json desc;
        type_to_json(desc, sdfg->type(container));
        j["containers"][container] = desc;
    }

    j["arguments"] = nlohmann::json::array();
    for (const auto& argument : sdfg->arguments()) {
        j["arguments"].push_back(argument);
    }

    j["externals"] = nlohmann::json::array();
    for (const auto& external : sdfg->externals()) {
        j["externals"].push_back(external);
    }

    j["metadata"] = nlohmann::json::object();
    for (const auto& entry : sdfg->metadata()) {
        j["metadata"][entry.first] = entry.second;
    }

    // Walk the SDFG
    nlohmann::json root_json;
    sequence_to_json(root_json, sdfg->root());
    j["root"] = root_json;

    return j;
}

void JSONSerializer::dataflow_to_json(nlohmann::json& j, const data_flow::DataFlowGraph& dataflow) {
    j["type"] = "dataflow";
    j["nodes"] = nlohmann::json::array();
    j["edges"] = nlohmann::json::array();

    for (auto& node : dataflow.nodes()) {
        nlohmann::json node_json;
        node_json["element_id"] = node.element_id();

        node_json["debug_info"] = nlohmann::json::object();
        debug_info_to_json(node_json["debug_info"], node.debug_info());

        if (auto code_node = dynamic_cast<const data_flow::Tasklet*>(&node)) {
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
            //     node_json["condition"] = dumps_expression(code_node->condition());
            // }
        } else if (auto lib_node = dynamic_cast<const data_flow::LibraryNode*>(&node)) {
            node_json["type"] = "library_node";
            node_json["call"] = lib_node->call();
            node_json["has_side_effect"] = lib_node->has_side_effect();
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
        } else if (auto code_node = dynamic_cast<const data_flow::AccessNode*>(&node)) {
            node_json["type"] = "access_node";
            node_json["data"] = code_node->data();
        } else {
            throw std::runtime_error("Unknown node type");
        }

        j["nodes"].push_back(node_json);
    }

    for (auto& edge : dataflow.edges()) {
        nlohmann::json edge_json;
        edge_json["element_id"] = edge.element_id();

        edge_json["debug_info"] = nlohmann::json::object();
        debug_info_to_json(edge_json["debug_info"], edge.debug_info());

        edge_json["src"] = edge.src().element_id();
        edge_json["dst"] = edge.dst().element_id();

        edge_json["src_conn"] = edge.src_conn();
        edge_json["dst_conn"] = edge.dst_conn();

        edge_json["subset"] = nlohmann::json::array();
        for (auto& subset : edge.subset()) {
            edge_json["subset"].push_back(expression(subset));
        }

        j["edges"].push_back(edge_json);
    }
}

void JSONSerializer::block_to_json(nlohmann::json& j, const structured_control_flow::Block& block) {
    j["type"] = "block";
    j["element_id"] = block.element_id();

    j["debug_info"] = nlohmann::json::object();
    debug_info_to_json(j["debug_info"], block.debug_info());

    nlohmann::json dataflow_json;
    dataflow_to_json(dataflow_json, block.dataflow());
    j["dataflow"] = dataflow_json;
}

void JSONSerializer::for_to_json(nlohmann::json& j, const structured_control_flow::For& for_node) {
    j["type"] = "for";
    j["element_id"] = for_node.element_id();

    j["debug_info"] = nlohmann::json::object();
    debug_info_to_json(j["debug_info"], for_node.debug_info());

    j["indvar"] = expression(for_node.indvar());
    j["init"] = expression(for_node.init());
    j["condition"] = expression(for_node.condition());
    j["update"] = expression(for_node.update());

    nlohmann::json body_json;
    sequence_to_json(body_json, for_node.root());
    j["root"] = body_json;
}

void JSONSerializer::if_else_to_json(nlohmann::json& j,
                                     const structured_control_flow::IfElse& if_else_node) {
    j["type"] = "if_else";
    j["element_id"] = if_else_node.element_id();

    j["debug_info"] = nlohmann::json::object();
    debug_info_to_json(j["debug_info"], if_else_node.debug_info());

    j["branches"] = nlohmann::json::array();
    for (size_t i = 0; i < if_else_node.size(); i++) {
        nlohmann::json branch_json;
        branch_json["condition"] = expression(if_else_node.at(i).second);
        nlohmann::json body_json;
        sequence_to_json(body_json, if_else_node.at(i).first);
        branch_json["root"] = body_json;
        j["branches"].push_back(branch_json);
    }
}

void JSONSerializer::while_node_to_json(nlohmann::json& j,
                                        const structured_control_flow::While& while_node) {
    j["type"] = "while";
    j["element_id"] = while_node.element_id();

    j["debug_info"] = nlohmann::json::object();
    debug_info_to_json(j["debug_info"], while_node.debug_info());

    nlohmann::json body_json;
    sequence_to_json(body_json, while_node.root());
    j["root"] = body_json;
}

void JSONSerializer::break_node_to_json(nlohmann::json& j,
                                        const structured_control_flow::Break& break_node) {
    j["type"] = "break";
    j["element_id"] = break_node.element_id();

    j["debug_info"] = nlohmann::json::object();
    debug_info_to_json(j["debug_info"], break_node.debug_info());
}

void JSONSerializer::continue_node_to_json(nlohmann::json& j,
                                           const structured_control_flow::Continue& continue_node) {
    j["type"] = "continue";
    j["element_id"] = continue_node.element_id();

    j["debug_info"] = nlohmann::json::object();
    debug_info_to_json(j["debug_info"], continue_node.debug_info());
}

void JSONSerializer::kernel_to_json(nlohmann::json& j,
                                    const structured_control_flow::Kernel& kernel_node) {
    j["type"] = "kernel";
    j["element_id"] = kernel_node.element_id();

    j["debug_info"] = nlohmann::json::object();
    debug_info_to_json(j["debug_info"], kernel_node.debug_info());

    j["suffix"] = kernel_node.suffix();

    nlohmann::json body_json;
    sequence_to_json(body_json, kernel_node.root());
    j["root"] = body_json;
}

void JSONSerializer::map_to_json(nlohmann::json& j, const structured_control_flow::Map& map_node) {
    j["type"] = "map";
    j["element_id"] = map_node.element_id();

    j["debug_info"] = nlohmann::json::object();
    debug_info_to_json(j["debug_info"], map_node.debug_info());

    j["indvar"] = expression(map_node.indvar());
    j["num_iterations"] = expression(map_node.num_iterations());
    nlohmann::json body_json;
    sequence_to_json(body_json, map_node.root());
    j["root"] = body_json;
}

void JSONSerializer::return_node_to_json(nlohmann::json& j,
                                         const structured_control_flow::Return& return_node) {
    j["type"] = "return";
    j["element_id"] = return_node.element_id();

    j["debug_info"] = nlohmann::json::object();
    debug_info_to_json(j["debug_info"], return_node.debug_info());
}

void JSONSerializer::sequence_to_json(nlohmann::json& j,
                                      const structured_control_flow::Sequence& sequence) {
    j["type"] = "sequence";
    j["element_id"] = sequence.element_id();

    j["debug_info"] = nlohmann::json::object();
    debug_info_to_json(j["debug_info"], sequence.debug_info());

    j["children"] = nlohmann::json::array();
    j["transitions"] = nlohmann::json::array();

    for (size_t i = 0; i < sequence.size(); i++) {
        nlohmann::json child_json;
        auto& child = sequence.at(i).first;
        auto& transition = sequence.at(i).second;

        if (auto block = dynamic_cast<const structured_control_flow::Block*>(&child)) {
            block_to_json(child_json, *block);
        } else if (auto for_node = dynamic_cast<const structured_control_flow::For*>(&child)) {
            for_to_json(child_json, *for_node);
        } else if (auto sequence_node =
                       dynamic_cast<const structured_control_flow::Sequence*>(&child)) {
            sequence_to_json(child_json, *sequence_node);
        } else if (auto condition_node =
                       dynamic_cast<const structured_control_flow::IfElse*>(&child)) {
            if_else_to_json(child_json, *condition_node);
        } else if (auto while_node = dynamic_cast<const structured_control_flow::While*>(&child)) {
            while_node_to_json(child_json, *while_node);
        } else if (auto kernel_node =
                       dynamic_cast<const structured_control_flow::Kernel*>(&child)) {
            kernel_to_json(child_json, *kernel_node);
        } else if (auto return_node =
                       dynamic_cast<const structured_control_flow::Return*>(&child)) {
            return_node_to_json(child_json, *return_node);
        } else if (auto break_node = dynamic_cast<const structured_control_flow::Break*>(&child)) {
            break_node_to_json(child_json, *break_node);
        } else if (auto continue_node =
                       dynamic_cast<const structured_control_flow::Continue*>(&child)) {
            continue_node_to_json(child_json, *continue_node);
        } else if (auto map_node = dynamic_cast<const structured_control_flow::Map*>(&child)) {
            map_to_json(child_json, *map_node);
        } else {
            throw std::runtime_error("Unknown child type");
        }

        j["children"].push_back(child_json);

        // Add transition information
        nlohmann::json transition_json;
        transition_json["type"] = "transition";
        transition_json["element_id"] = transition.element_id();

        transition_json["debug_info"] = nlohmann::json::object();
        debug_info_to_json(transition_json["debug_info"], transition.debug_info());

        transition_json["assignments"] = nlohmann::json::array();
        for (const auto& assignment : transition.assignments()) {
            nlohmann::json assignment_json;
            assignment_json["symbol"] = expression(assignment.first);
            assignment_json["expression"] = expression(assignment.second);
            transition_json["assignments"].push_back(assignment_json);
        }

        j["transitions"].push_back(transition_json);
    }
}

void JSONSerializer::type_to_json(nlohmann::json& j, const types::IType& type) {
    if (auto scalar_type = dynamic_cast<const types::Scalar*>(&type)) {
        j["type"] = "scalar";
        j["primitive_type"] = scalar_type->primitive_type();
        j["address_space"] = scalar_type->address_space();
        j["initializer"] = scalar_type->initializer();
        j["device_location"] = scalar_type->device_location();
    } else if (auto array_type = dynamic_cast<const types::Array*>(&type)) {
        j["type"] = "array";
        nlohmann::json element_type_json;
        type_to_json(element_type_json, array_type->element_type());
        j["element_type"] = element_type_json;
        j["num_elements"] = expression(array_type->num_elements());
        j["address_space"] = array_type->address_space();
        j["initializer"] = array_type->initializer();
        j["device_location"] = array_type->device_location();
        j["alignment"] = array_type->alignment();
    } else if (auto pointer_type = dynamic_cast<const types::Pointer*>(&type)) {
        j["type"] = "pointer";
        nlohmann::json pointee_type_json;
        type_to_json(pointee_type_json, pointer_type->pointee_type());
        j["pointee_type"] = pointee_type_json;
        j["address_space"] = pointer_type->address_space();
        j["initializer"] = pointer_type->initializer();
        j["device_location"] = pointer_type->device_location();
    } else if (auto structure_type = dynamic_cast<const types::Structure*>(&type)) {
        j["type"] = "structure";
        j["name"] = structure_type->name();
        j["address_space"] = structure_type->address_space();
        j["initializer"] = structure_type->initializer();
        j["device_location"] = structure_type->device_location();
    } else if (auto function_type = dynamic_cast<const types::Function*>(&type)) {
        j["type"] = "function";
        nlohmann::json return_type_json;
        type_to_json(return_type_json, function_type->return_type());
        j["return_type"] = return_type_json;
        j["params"] = nlohmann::json::array();
        for (size_t i = 0; i < function_type->num_params(); i++) {
            nlohmann::json param_json;
            type_to_json(param_json, function_type->param_type(symbolic::integer(i)));
            j["params"].push_back(param_json);
        }
        j["is_var_arg"] = function_type->is_var_arg();
        j["address_space"] = function_type->address_space();
        j["initializer"] = function_type->initializer();
        j["device_location"] = function_type->device_location();
    } else {
        throw std::runtime_error("Unknown type");
    }
}

void JSONSerializer::structure_definition_to_json(nlohmann::json& j,
                                                  const types::StructureDefinition& definition) {
    j["name"] = definition.name();
    j["members"] = nlohmann::json::array();
    for (size_t i = 0; i < definition.num_members(); i++) {
        nlohmann::json member_json;
        type_to_json(member_json, definition.member_type(symbolic::integer(i)));
        j["members"].push_back(member_json);
    }
    j["is_packed"] = definition.is_packed();
}

void JSONSerializer::debug_info_to_json(nlohmann::json& j, const DebugInfo& debug_info) {
    j["has"] = debug_info.has();
    j["filename"] = debug_info.filename();
    j["start_line"] = debug_info.start_line();
    j["start_column"] = debug_info.start_column();
    j["end_line"] = debug_info.end_line();
    j["end_column"] = debug_info.end_column();
}

/*
 * * Deserialization logic
 */

std::unique_ptr<StructuredSDFG> JSONSerializer::deserialize(nlohmann::json& j) {
    assert(j.contains("name"));
    assert(j["name"].is_string());

    builder::StructuredSDFGBuilder builder(j["name"]);

    // deserialize structures
    assert(j.contains("structures"));
    assert(j["structures"].is_array());
    for (const auto& structure : j["structures"]) {
        assert(structure.contains("name"));
        assert(structure["name"].is_string());
        json_to_structure_definition(structure, builder);
    }

    nlohmann::json& containers = j["containers"];

    // deserialize externals
    for (const auto& name : j["externals"]) {
        auto& type_desc = containers.at(name.get<std::string>());
        auto type = json_to_type(type_desc);
        builder.add_container(name, *type, false, true);
    }

    // deserialize arguments
    for (const auto& name : j["arguments"]) {
        auto& type_desc = containers.at(name.get<std::string>());
        auto type = json_to_type(type_desc);
        builder.add_container(name, *type, true, false);
    }

    // deserialize transients
    for (const auto& entry : containers.items()) {
        if (builder.subject().is_argument(entry.key())) {
            continue;
        }
        if (builder.subject().is_external(entry.key())) {
            continue;
        }
        auto type = json_to_type(entry.value());
        builder.add_container(entry.key(), *type, false, false);
    }

    // deserialize root node
    assert(j.contains("root"));
    auto& root = builder.subject().root();
    json_to_sequence(j["root"], builder, root);

    // deserialize metadata
    assert(j.contains("metadata"));
    assert(j["metadata"].is_object());
    for (const auto& entry : j["metadata"].items()) {
        builder.subject().add_metadata(entry.key(), entry.value());
    }

    return builder.move();
}

void JSONSerializer::json_to_structure_definition(const nlohmann::json& j,
                                                  builder::StructuredSDFGBuilder& builder) {
    assert(j.contains("name"));
    assert(j["name"].is_string());
    assert(j.contains("members"));
    assert(j["members"].is_array());
    assert(j.contains("is_packed"));
    assert(j["is_packed"].is_boolean());
    auto is_packed = j["is_packed"];
    auto& definition = builder.add_structure(j["name"], is_packed);
    for (const auto& member : j["members"]) {
        nlohmann::json member_json;
        auto member_type = json_to_type(member);
        definition.add_member(*member_type);
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
                                      builder::StructuredSDFGBuilder& builder,
                                      structured_control_flow::Block& parent) {
    std::unordered_map<std::string, data_flow::DataFlowNode&> nodes_map;

    assert(j.contains("nodes"));
    assert(j["nodes"].is_array());
    for (const auto& node : j["nodes"]) {
        assert(node.contains("type"));
        assert(node["type"].is_string());
        assert(node.contains("element_id"));
        assert(node["element_id"].is_string());
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
            auto& tasklet = builder.add_tasklet(parent, node["code"], outputs.at(0), inputs,
                                                json_to_debug_info(node["debug_info"]));
            tasklet.element_id_ = node["element_id"];
            nodes_map.insert({node["element_id"], tasklet});
        } else if (type == "library_node") {
            assert(node.contains("call"));
            assert(node.contains("inputs"));
            assert(node["inputs"].is_array());
            assert(node.contains("outputs"));
            assert(node["outputs"].is_array());
            auto outputs = json_to_arguments(node["outputs"]);
            auto inputs = json_to_arguments(node["inputs"]);
            auto& lib_node = builder.add_library_node(parent, node["call"], outputs, inputs,
                                                      node["has_side_effect"],
                                                      json_to_debug_info(node["debug_info"]));
            lib_node.element_id_ = node["element_id"];
            nodes_map.insert({node["element_id"], lib_node});
        } else if (type == "access_node") {
            assert(node.contains("data"));
            auto& access_node =
                builder.add_access(parent, node["data"], json_to_debug_info(node["debug_info"]));
            access_node.element_id_ = node["element_id"];
            nodes_map.insert({node["element_id"], access_node});
        } else {
            throw std::runtime_error("Unknown node type");
        }
    }

    assert(j.contains("edges"));
    assert(j["edges"].is_array());
    for (const auto& edge : j["edges"]) {
        assert(edge.contains("src"));
        assert(edge["src"].is_string());
        assert(edge.contains("dst"));
        assert(edge["dst"].is_string());
        assert(edge.contains("src_conn"));
        assert(edge["src_conn"].is_string());
        assert(edge.contains("dst_conn"));
        assert(edge["dst_conn"].is_string());
        assert(edge.contains("subset"));
        assert(edge["subset"].is_array());

        assert(nodes_map.find(edge["src"]) != nodes_map.end());
        assert(nodes_map.find(edge["dst"]) != nodes_map.end());
        auto& source = nodes_map.at(edge["src"]);
        auto& target = nodes_map.at(edge["dst"]);

        assert(edge.contains("subset"));
        assert(edge["subset"].is_array());
        std::vector<symbolic::Expression> subset;
        for (const auto& subset_str : edge["subset"]) {
            assert(subset_str.is_string());
            SymEngine::Expression subset_expr(subset_str);
            subset.push_back(subset_expr);
        }
        auto& memlet =
            builder.add_memlet(parent, source, edge["src_conn"], target, edge["dst_conn"], subset,
                               json_to_debug_info(edge["debug_info"]));
        memlet.element_id_ = edge["element_id"];
    }
}

void JSONSerializer::json_to_sequence(const nlohmann::json& j,
                                      builder::StructuredSDFGBuilder& builder,
                                      structured_control_flow::Sequence& sequence) {
    assert(j.contains("type"));
    assert(j["type"].is_string());
    assert(j.contains("children"));
    assert(j["children"].is_array());
    assert(j.contains("transitions"));
    assert(j["transitions"].is_array());
    assert(j["transitions"].size() == j["children"].size());

    sequence.element_id_ = j["element_id"];
    sequence.debug_info_ = json_to_debug_info(j["debug_info"]);

    std::string type = j["type"];
    if (type == "sequence") {
        for (size_t i = 0; i < j["children"].size(); i++) {
            auto& child = j["children"][i];
            auto& transition = j["transitions"][i];
            assert(child.contains("type"));
            assert(child["type"].is_string());

            assert(transition.contains("type"));
            assert(transition["type"].is_string());
            assert(transition.contains("assignments"));
            assert(transition["assignments"].is_array());
            symbolic::Assignments assignments;
            for (const auto& assignment : transition["assignments"]) {
                assert(assignment.contains("symbol"));
                assert(assignment["symbol"].is_string());
                assert(assignment.contains("expression"));
                assert(assignment["expression"].is_string());
                SymEngine::Expression expr(assignment["expression"]);
                assignments.insert({symbolic::symbol(assignment["symbol"]), expr});
            }

            if (child["type"] == "block") {
                json_to_block_node(child, builder, sequence, assignments);
            } else if (child["type"] == "for") {
                json_to_for_node(child, builder, sequence, assignments);
            } else if (child["type"] == "if_else") {
                json_to_if_else_node(child, builder, sequence);
            } else if (child["type"] == "while") {
                json_to_while_node(child, builder, sequence, assignments);
            } else if (child["type"] == "break") {
                json_to_break_node(child, builder, sequence);
            } else if (child["type"] == "continue") {
                json_to_continue_node(child, builder, sequence);
            } else if (child["type"] == "kernel") {
                json_to_kernel_node(child, builder, sequence);
            } else if (child["type"] == "return") {
                json_to_return_node(child, builder, sequence, assignments);
            } else {
                throw std::runtime_error("Unknown child type");
            }

            sequence.at(i).second.debug_info_ = json_to_debug_info(transition["debug_info"]);
            sequence.at(i).second.element_id_ = transition["element_id"];
        }
    } else {
        throw std::runtime_error("expected sequence type");
    }
}

void JSONSerializer::json_to_block_node(const nlohmann::json& j,
                                        builder::StructuredSDFGBuilder& builder,
                                        structured_control_flow::Sequence& parent,
                                        symbolic::Assignments& assignments) {
    assert(j.contains("type"));
    assert(j["type"].is_string());
    assert(j.contains("dataflow"));
    assert(j["dataflow"].is_object());
    auto& block = builder.add_block(parent, assignments, json_to_debug_info(j["debug_info"]));
    block.element_id_ = j["element_id"];
    assert(j["dataflow"].contains("type"));
    assert(j["dataflow"]["type"].is_string());
    std::string type = j["dataflow"]["type"];
    if (type == "dataflow") {
        json_to_dataflow(j["dataflow"], builder, block);
    } else {
        throw std::runtime_error("Unknown dataflow type");
    }
}

void JSONSerializer::json_to_for_node(const nlohmann::json& j,
                                      builder::StructuredSDFGBuilder& builder,
                                      structured_control_flow::Sequence& parent,
                                      symbolic::Assignments& assignments) {
    assert(j.contains("type"));
    assert(j["type"].is_string());
    assert(j.contains("indvar"));
    assert(j["indvar"].is_string());
    assert(j.contains("init"));
    assert(j["init"].is_string());
    assert(j.contains("condition"));
    assert(j["condition"].is_string());
    assert(j.contains("update"));
    assert(j["update"].is_string());
    assert(j.contains("root"));
    assert(j["root"].is_object());

    symbolic::Symbol indvar = symbolic::symbol(j["indvar"]);
    SymEngine::Expression init(j["init"]);
    SymEngine::Expression condition_expr(j["condition"]);
    assert(!SymEngine::rcp_static_cast<const SymEngine::Boolean>(condition_expr.get_basic())
                .is_null());
    symbolic::Condition condition =
        SymEngine::rcp_static_cast<const SymEngine::Boolean>(condition_expr.get_basic());
    SymEngine::Expression update(j["update"]);
    auto& for_node = builder.add_for(parent, indvar, condition, init, update, assignments,
                                     json_to_debug_info(j["debug_info"]));
    for_node.element_id_ = j["element_id"];

    assert(j["root"].contains("type"));
    assert(j["root"]["type"].is_string());
    assert(j["root"]["type"] == "sequence");
    json_to_sequence(j["root"], builder, for_node.root());
}

void JSONSerializer::json_to_if_else_node(const nlohmann::json& j,
                                          builder::StructuredSDFGBuilder& builder,
                                          structured_control_flow::Sequence& parent) {
    assert(j.contains("type"));
    assert(j["type"].is_string());
    assert(j["type"] == "if_else");
    assert(j.contains("branches"));
    assert(j["branches"].is_array());
    auto& if_else_node = builder.add_if_else(parent, json_to_debug_info(j["debug_info"]));
    if_else_node.element_id_ = j["element_id"];
    for (const auto& branch : j["branches"]) {
        assert(branch.contains("condition"));
        assert(branch["condition"].is_string());
        assert(branch.contains("root"));
        assert(branch["root"].is_object());
        SymEngine::Expression condition_expr(branch["condition"]);
        assert(!SymEngine::rcp_static_cast<const SymEngine::Boolean>(condition_expr.get_basic())
                    .is_null());
        symbolic::Condition condition =
            SymEngine::rcp_static_cast<const SymEngine::Boolean>(condition_expr.get_basic());
        auto& branch_node = builder.add_case(if_else_node, condition);
        assert(branch["root"].contains("type"));
        assert(branch["root"]["type"].is_string());
        std::string type = branch["root"]["type"];
        if (type == "sequence") {
            json_to_sequence(branch["root"], builder, branch_node);
        } else {
            throw std::runtime_error("Unknown child type");
        }
    }
}

void JSONSerializer::json_to_while_node(const nlohmann::json& j,
                                        builder::StructuredSDFGBuilder& builder,
                                        structured_control_flow::Sequence& parent,
                                        symbolic::Assignments& assignments) {
    assert(j.contains("type"));
    assert(j["type"].is_string());
    assert(j["type"] == "while");
    assert(j.contains("root"));
    assert(j["root"].is_object());

    auto& while_node = builder.add_while(parent, assignments, json_to_debug_info(j["debug_info"]));
    while_node.element_id_ = j["element_id"];

    assert(j["root"]["type"] == "sequence");
    json_to_sequence(j["root"], builder, while_node.root());
}

void JSONSerializer::json_to_break_node(const nlohmann::json& j,
                                        builder::StructuredSDFGBuilder& builder,
                                        structured_control_flow::Sequence& parent) {
    assert(j.contains("type"));
    assert(j["type"].is_string());
    assert(j["type"] == "break");
    auto& node = builder.add_break(parent, json_to_debug_info(j["debug_info"]));
    node.element_id_ = j["element_id"];
}

void JSONSerializer::json_to_continue_node(const nlohmann::json& j,
                                           builder::StructuredSDFGBuilder& builder,
                                           structured_control_flow::Sequence& parent) {
    assert(j.contains("type"));
    assert(j["type"].is_string());
    assert(j["type"] == "continue");
    auto& node = builder.add_continue(parent, json_to_debug_info(j["debug_info"]));
    node.element_id_ = j["element_id"];
}

void JSONSerializer::json_to_kernel_node(const nlohmann::json& j,
                                         builder::StructuredSDFGBuilder& builder,
                                         structured_control_flow::Sequence& parent) {
    assert(j.contains("type"));
    assert(j["type"].is_string());
    assert(j["type"] == "kernel");
    assert(j.contains("suffix"));
    assert(j["suffix"].is_string());
    assert(j.contains("root"));
    assert(j["root"].is_object());
    auto& kernel_node =
        builder.add_kernel(parent, j["suffix"], json_to_debug_info(j["debug_info"]));
    kernel_node.element_id_ = j["element_id"];

    assert(j["root"].contains("type"));
    assert(j["root"]["type"].is_string());
    assert(j["root"]["type"] == "sequence");
    json_to_sequence(j["root"], builder, kernel_node.root());
}

void JSONSerializer::json_to_map_node(const nlohmann::json& j,
                                      builder::StructuredSDFGBuilder& builder,
                                      structured_control_flow::Sequence& parent,
                                      symbolic::Assignments& assignments) {
    assert(j.contains("type"));
    assert(j["type"].is_string());
    assert(j["type"] == "map");
    assert(j.contains("indvar"));
    assert(j["indvar"].is_string());
    assert(j.contains("num_iterations"));
    assert(j["num_iterations"].is_string());
    assert(j.contains("root"));
    assert(j["root"].is_object());
    symbolic::Symbol indvar = symbolic::symbol(j["indvar"]);
    SymEngine::Expression num_iterations(j["num_iterations"]);

    auto& map_node = builder.add_map(parent, indvar, num_iterations, assignments,
                                     json_to_debug_info(j["debug_info"]));
    map_node.element_id_ = j["element_id"];

    assert(j["root"].contains("type"));
    assert(j["root"]["type"].is_string());
    assert(j["root"]["type"] == "sequence");
    json_to_sequence(j["root"], builder, map_node.root());
}

void JSONSerializer::json_to_return_node(const nlohmann::json& j,
                                         builder::StructuredSDFGBuilder& builder,
                                         structured_control_flow::Sequence& parent,
                                         symbolic::Assignments& assignments) {
    assert(j.contains("type"));
    assert(j["type"].is_string());
    assert(j["type"] == "return");

    auto& node = builder.add_return(parent, assignments, json_to_debug_info(j["debug_info"]));
    node.element_id_ = j["element_id"];
}

std::unique_ptr<types::IType> JSONSerializer::json_to_type(const nlohmann::json& j) {
    if (j.contains("type")) {
        if (j["type"] == "scalar") {
            // Deserialize scalar type
            assert(j.contains("primitive_type"));
            types::PrimitiveType primitive_type = j["primitive_type"];
            assert(j.contains("device_location"));
            types::DeviceLocation device_location = j["device_location"];
            assert(j.contains("address_space"));
            uint address_space = j["address_space"];
            assert(j.contains("initializer"));
            std::string initializer = j["initializer"];
            return std::make_unique<types::Scalar>(primitive_type, device_location, address_space,
                                                   initializer);
        } else if (j["type"] == "array") {
            // Deserialize array type
            assert(j.contains("element_type"));
            std::unique_ptr<types::IType> member_type = json_to_type(j["element_type"]);
            assert(j.contains("num_elements"));
            std::string num_elements_str = j["num_elements"];
            // Convert num_elements_str to symbolic::Expression
            SymEngine::Expression num_elements(num_elements_str);
            assert(j.contains("address_space"));
            uint address_space = j["address_space"];
            assert(j.contains("initializer"));
            std::string initializer = j["initializer"];
            assert(j.contains("device_location"));
            types::DeviceLocation device_location = j["device_location"];
            assert(j.contains("alignment"));
            size_t alignment = j["alignment"];
            return std::make_unique<types::Array>(*member_type, num_elements, device_location,
                                                  address_space, initializer, alignment);
        } else if (j["type"] == "pointer") {
            // Deserialize pointer type
            assert(j.contains("pointee_type"));
            std::unique_ptr<types::IType> pointee_type = json_to_type(j["pointee_type"]);
            assert(j.contains("address_space"));
            uint address_space = j["address_space"];
            assert(j.contains("initializer"));
            std::string initializer = j["initializer"];
            assert(j.contains("device_location"));
            types::DeviceLocation device_location = j["device_location"];
            return std::make_unique<types::Pointer>(*pointee_type, device_location, address_space,
                                                    initializer);
        } else if (j["type"] == "structure") {
            // Deserialize structure type
            assert(j.contains("name"));
            std::string name = j["name"];
            assert(j.contains("address_space"));
            uint address_space = j["address_space"];
            assert(j.contains("initializer"));
            std::string initializer = j["initializer"];
            assert(j.contains("device_location"));
            types::DeviceLocation device_location = j["device_location"];
            return std::make_unique<types::Structure>(name, device_location, address_space,
                                                      initializer);
        } else if (j["type"] == "function") {
            // Deserialize function type
            assert(j.contains("return_type"));
            std::unique_ptr<types::IType> return_type = json_to_type(j["return_type"]);
            assert(j.contains("params"));
            std::vector<std::unique_ptr<types::IType>> params;
            for (const auto& param : j["params"]) {
                params.push_back(json_to_type(param));
            }
            assert(j.contains("is_var_arg"));
            bool is_var_arg = j["is_var_arg"];
            assert(j.contains("address_space"));
            uint address_space = j["address_space"];
            assert(j.contains("initializer"));
            std::string initializer = j["initializer"];
            assert(j.contains("device_location"));
            types::DeviceLocation device_location = j["device_location"];
            auto function = std::make_unique<types::Function>(
                *return_type, is_var_arg, device_location, address_space, initializer);
            for (const auto& param : params) {
                function->add_param(*param);
            }
            return function->clone();

        } else {
            throw std::runtime_error("Unknown type");
        }
    } else {
        throw std::runtime_error("Type not found");
    }
}

DebugInfo JSONSerializer::json_to_debug_info(const nlohmann::json& j) {
    assert(j.contains("has"));
    assert(j["has"].is_boolean());
    if (!j["has"]) {
        return DebugInfo();
    }
    assert(j.contains("filename"));
    assert(j["filename"].is_string());
    std::string filename = j["filename"];
    assert(j.contains("start_line"));
    assert(j["start_line"].is_number_integer());
    size_t start_line = j["start_line"];
    assert(j.contains("start_column"));
    assert(j["start_column"].is_number_integer());
    size_t start_column = j["start_column"];
    assert(j.contains("end_line"));
    assert(j["end_line"].is_number_integer());
    size_t end_line = j["end_line"];
    assert(j.contains("end_column"));
    assert(j["end_column"].is_number_integer());
    size_t end_column = j["end_column"];
    return DebugInfo(filename, start_line, start_column, end_line, end_column);
}

std::string JSONSerializer::expression(const symbolic::Expression& expr) {
    JSONSymbolicPrinter printer;
    return printer.apply(expr);
};

void JSONSymbolicPrinter::bvisit(const SymEngine::Equality& x) {
    str_ = apply(x.get_args()[0]) + " == " + apply(x.get_args()[1]);
    str_ = parenthesize(str_);
};

void JSONSymbolicPrinter::bvisit(const SymEngine::Unequality& x) {
    str_ = apply(x.get_args()[0]) + " != " + apply(x.get_args()[1]);
    str_ = parenthesize(str_);
};

void JSONSymbolicPrinter::bvisit(const SymEngine::LessThan& x) {
    str_ = apply(x.get_args()[0]) + " <= " + apply(x.get_args()[1]);
    str_ = parenthesize(str_);
};

void JSONSymbolicPrinter::bvisit(const SymEngine::StrictLessThan& x) {
    str_ = apply(x.get_args()[0]) + " < " + apply(x.get_args()[1]);
    str_ = parenthesize(str_);
};

void JSONSymbolicPrinter::bvisit(const SymEngine::Min& x) {
    std::ostringstream s;
    auto container = x.get_args();
    if (container.size() == 1) {
        s << apply(*container.begin());
    } else {
        s << "min(";
        s << apply(*container.begin());

        // Recursively apply __daisy_min to the arguments
        SymEngine::vec_basic subargs;
        for (auto it = ++(container.begin()); it != container.end(); ++it) {
            subargs.push_back(*it);
        }
        auto submin = SymEngine::min(subargs);
        s << ", " << apply(submin);

        s << ")";
    }

    str_ = s.str();
};

void JSONSymbolicPrinter::bvisit(const SymEngine::Max& x) {
    std::ostringstream s;
    auto container = x.get_args();
    if (container.size() == 1) {
        s << apply(*container.begin());
    } else {
        s << "max(";
        s << apply(*container.begin());

        // Recursively apply __daisy_max to the arguments
        SymEngine::vec_basic subargs;
        for (auto it = ++(container.begin()); it != container.end(); ++it) {
            subargs.push_back(*it);
        }
        auto submax = SymEngine::max(subargs);
        s << ", " << apply(submax);

        s << ")";
    }

    str_ = s.str();
};

}  // namespace serializer
}  // namespace sdfg
