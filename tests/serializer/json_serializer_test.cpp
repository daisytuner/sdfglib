#include "sdfg/serializer/json_serializer.h"

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/element.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/return.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/while.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/function.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"
#include "symengine/expression.h"

using namespace sdfg;

TEST(JSONSerializerTest, DatatypeToJSON_Scalar) {
    // Create a sample data type
    types::Scalar scalar_type(types::PrimitiveType::Int32);
    nlohmann::json j;

    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the data type to JSON
    serializer.type_to_json(j, scalar_type);

    // Check if the JSON contains the expected keys
    EXPECT_TRUE(j.contains("type"));
    EXPECT_EQ(j["type"], "scalar");
    EXPECT_TRUE(j.contains("primitive_type"));
    EXPECT_EQ(j["primitive_type"], scalar_type.primitive_type());
    EXPECT_TRUE(j.contains("storage_type"));
    EXPECT_EQ(j["storage_type"].get<std::string_view>(), scalar_type.storage_type().value());
    EXPECT_TRUE(j.contains("alignment"));
    EXPECT_EQ(j["alignment"], scalar_type.alignment());
    EXPECT_TRUE(j.contains("initializer"));
    EXPECT_EQ(j["initializer"], scalar_type.initializer());
}

TEST(JSONSerializerTest, DatatypeToJSON_Pointer) {
    // Create a sample data type
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer pointer_type(base_desc);
    nlohmann::json j;

    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the data type to JSON
    serializer.type_to_json(j, pointer_type);

    // Check if the JSON contains the expected keys
    EXPECT_TRUE(j.contains("type"));
    EXPECT_EQ(j["type"], "pointer");
    EXPECT_TRUE(j.contains("pointee_type"));
    EXPECT_EQ(j["pointee_type"]["type"], "scalar");
    EXPECT_EQ(j["pointee_type"]["primitive_type"], base_desc.primitive_type());
    EXPECT_TRUE(j.contains("storage_type"));
    EXPECT_EQ(j["storage_type"].get<std::string_view>(), pointer_type.storage_type().value());
    EXPECT_TRUE(j.contains("alignment"));
    EXPECT_EQ(j["alignment"], pointer_type.alignment());
    EXPECT_TRUE(j.contains("initializer"));
    EXPECT_EQ(j["initializer"], pointer_type.initializer());
}

TEST(JSONSerializerTest, DatatypeToJSON_Structure) {
    // Create a sample data type
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Structure structure_type("MyStruct");
    nlohmann::json j;

    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the data type to JSON
    serializer.type_to_json(j, structure_type);

    // Check if the JSON contains the expected keys
    EXPECT_TRUE(j.contains("type"));
    EXPECT_EQ(j["type"], "structure");
    EXPECT_TRUE(j.contains("name"));
    EXPECT_EQ(j["name"], "MyStruct");
    EXPECT_TRUE(j.contains("storage_type"));
    EXPECT_EQ(j["storage_type"].get<std::string_view>(), structure_type.storage_type().value());
    EXPECT_TRUE(j.contains("alignment"));
    EXPECT_EQ(j["alignment"], structure_type.alignment());
    EXPECT_TRUE(j.contains("initializer"));
    EXPECT_EQ(j["initializer"], structure_type.initializer());
}

TEST(JSONSerializerTest, DatatypeToJSON_Array) {
    // Create a sample data type
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_type(base_desc, {symbolic::symbol("N")});
    nlohmann::json j;

    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the data type to JSON
    serializer.type_to_json(j, array_type);

    // Check if the JSON contains the expected keys
    EXPECT_TRUE(j.contains("type"));
    EXPECT_EQ(j["type"], "array");
    EXPECT_TRUE(j.contains("element_type"));
    EXPECT_EQ(j["element_type"]["type"], "scalar");
    EXPECT_EQ(j["element_type"]["primitive_type"], base_desc.primitive_type());
    EXPECT_TRUE(j.contains("num_elements"));
    EXPECT_TRUE(symbolic::eq(SymEngine::Expression(j["num_elements"]), symbolic::symbol("N")));
    EXPECT_TRUE(j.contains("storage_type"));
    EXPECT_EQ(j["storage_type"].get<std::string_view>(), array_type.storage_type().value());
    EXPECT_TRUE(j.contains("alignment"));
    EXPECT_EQ(j["alignment"], array_type.alignment());
    EXPECT_TRUE(j.contains("initializer"));
    EXPECT_EQ(j["initializer"], array_type.initializer());
}

TEST(JSONSerializerTest, DatatypeToJSON_Function) {
    // Create a sample data type
    types::Scalar scalar_type(types::PrimitiveType::Int32);
    types::Function function_type(scalar_type, true);
    function_type.add_param(scalar_type);
    nlohmann::json j;

    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the data type to JSON
    serializer.type_to_json(j, function_type);

    // Check if the JSON contains the expected keys
    EXPECT_TRUE(j.contains("type"));
    EXPECT_EQ(j["type"], "function");
    EXPECT_TRUE(j.contains("return_type"));
    EXPECT_EQ(j["return_type"]["type"], "scalar");
    EXPECT_EQ(j["return_type"]["primitive_type"], scalar_type.primitive_type());
    EXPECT_TRUE(j.contains("params"));
    EXPECT_EQ(j["params"].size(), 1);
    EXPECT_EQ(j["params"][0]["type"], "scalar");
    EXPECT_EQ(j["params"][0]["primitive_type"], scalar_type.primitive_type());
    EXPECT_TRUE(j.contains("is_var_arg"));
    EXPECT_EQ(j["is_var_arg"], function_type.is_var_arg());
    EXPECT_TRUE(j.contains("storage_type"));
    EXPECT_EQ(j["storage_type"].get<std::string_view>(), scalar_type.storage_type().value());
    EXPECT_TRUE(j.contains("alignment"));
    EXPECT_EQ(j["alignment"], scalar_type.alignment());
    EXPECT_TRUE(j.contains("initializer"));
    EXPECT_EQ(j["initializer"], scalar_type.initializer());
}

TEST(JSONSerializerTest, DataflowToJSON) {
    // Create a sample Block object
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& block = builder.add_block(builder.subject().root());

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("D", desc, true);
    builder.add_container("C", base_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    auto& access_in = builder.add_access(block, "A");
    auto& access_in2 = builder.add_access(block, "C");
    auto& access_out = builder.add_access(block, "D");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::add, {"_out", base_desc},
                                        {{"_in1", base_desc}, {"_in2", base_desc}});
    auto& memlet_in =
        builder.add_memlet(block, access_in, "void", tasklet, "_in1", {{symbolic::symbol("i")}});
    auto& memlet_in2 = builder.add_memlet(block, access_in2, "void", tasklet, "_in2", {});
    auto& memlet_out =
        builder.add_memlet(block, tasklet, "_out", access_out, "void", {symbolic::symbol("i")});

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the Block to JSON
    nlohmann::json j;
    serializer.dataflow_to_json(j, block.dataflow());

    // Check if the JSON contains the expected keys
    EXPECT_TRUE(j.contains("type"));
    EXPECT_EQ(j["type"], "dataflow");
    EXPECT_TRUE(j.contains("nodes"));
    EXPECT_TRUE(j.contains("edges"));
    EXPECT_EQ(j["nodes"].size(), 4);
    EXPECT_EQ(j["edges"].size(), 3);

    // Check if the nodes and edges are serialized correctly
    auto access_node_A = std::find_if(j["nodes"].begin(), j["nodes"].end(), [](const auto& node) {
        return node["type"] == "access_node" && node["data"] == "A";
    });
    auto access_node_C = std::find_if(j["nodes"].begin(), j["nodes"].end(), [](const auto& node) {
        return node["type"] == "access_node" && node["data"] == "C";
    });
    auto access_node_D = std::find_if(j["nodes"].begin(), j["nodes"].end(), [](const auto& node) {
        return node["type"] == "access_node" && node["data"] == "D";
    });
    auto tasklet_node = std::find_if(j["nodes"].begin(), j["nodes"].end(), [](const auto& node) {
        return node["type"] == "tasklet" && node["code"] == data_flow::TaskletCode::add;
    });

    EXPECT_NE(access_node_A, j["nodes"].end());
    EXPECT_NE(access_node_C, j["nodes"].end());
    EXPECT_NE(access_node_D, j["nodes"].end());
    EXPECT_NE(tasklet_node, j["nodes"].end());

    EXPECT_EQ(tasklet_node->at("inputs").size(), 2);
    EXPECT_EQ(tasklet_node->at("output")["name"], "_out");
    EXPECT_EQ(tasklet_node->at("output")["type"]["type"], "scalar");
    EXPECT_EQ(tasklet_node->at("output")["type"]["primitive_type"], base_desc.primitive_type());

    auto edge_to_tasklet =
        std::find_if(j["edges"].begin(), j["edges"].end(), [&](const auto& edge) {
            return edge["src"] == access_node_A->at("element_id") &&
                   edge["dst"] == tasklet_node->at("element_id") && edge["src_conn"] == "void" &&
                   edge["dst_conn"] == "_in1";
        });
    auto edge_to_tasklet2 =
        std::find_if(j["edges"].begin(), j["edges"].end(), [&](const auto& edge) {
            return edge["src"] == access_node_C->at("element_id") &&
                   edge["dst"] == tasklet_node->at("element_id") && edge["src_conn"] == "void" &&
                   edge["dst_conn"] == "_in2";
        });
    auto edge_from_tasklet =
        std::find_if(j["edges"].begin(), j["edges"].end(), [&](const auto& edge) {
            return edge["src"] == tasklet_node->at("element_id") &&
                   edge["dst"] == access_node_D->at("element_id") && edge["src_conn"] == "_out" &&
                   edge["dst_conn"] == "void";
        });

    EXPECT_NE(edge_to_tasklet, j["edges"].end());
    EXPECT_NE(edge_to_tasklet2, j["edges"].end());
    EXPECT_NE(edge_from_tasklet, j["edges"].end());
}

TEST(JSONSerializerTest, BlockToJSON) {
    // Create a sample Block object
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& block = builder.add_block(builder.subject().root());

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the Block to JSON
    nlohmann::json j;
    serializer.block_to_json(j, block);

    // Check if the JSON contains the expected keys
    EXPECT_TRUE(j.contains("type"));
    EXPECT_EQ(j["type"], "block");
}

TEST(JSONSerializerTest, ForNodeToJSON) {
    // Create a sample For node
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    auto& scope = builder.add_for(
        root, symbolic::symbol("i"), symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0), symbolic::add(symbolic::symbol("i"), symbolic::integer(1)));
    auto& body = builder.add_block(scope.root());

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the For node to JSON
    nlohmann::json j;
    serializer.for_to_json(j, scope);

    // Check if the JSON contains the expected keys
    EXPECT_TRUE(j.contains("type"));
    EXPECT_EQ(j["type"], "for");

    EXPECT_TRUE(j.contains("indvar"));
    EXPECT_EQ(j["indvar"], "i");
    EXPECT_TRUE(j.contains("condition"));
    EXPECT_EQ(j["condition"], "(i < 10)");
    EXPECT_TRUE(j.contains("init"));
    EXPECT_EQ(j["init"], "0");
    EXPECT_TRUE(j.contains("update"));
    EXPECT_TRUE(symbolic::eq(SymEngine::Expression(j["update"]),
                             symbolic::add(symbolic::symbol("i"), symbolic::integer(1))));
    EXPECT_TRUE(j.contains("root"));
    EXPECT_EQ(j["root"]["type"], "sequence");
    EXPECT_EQ(j["root"]["children"].size(), 1);
    EXPECT_EQ(j["root"]["children"][0]["type"], "block");
}

TEST(JSONSerializerTest, IfElseToJSON) {
    // Create a sample IfElse node
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    auto& if_else = builder.add_if_else(root);
    auto& true_case = builder.add_case(if_else, symbolic::__true__());
    auto& false_case = builder.add_case(if_else, symbolic::__false__());
    auto& true_block = builder.add_block(true_case);
    auto& false_block = builder.add_block(false_case);

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the IfElse node to JSON
    nlohmann::json j;
    serializer.if_else_to_json(j, if_else);

    // Check if the JSON contains the expected keys
    EXPECT_TRUE(j.contains("type"));
    EXPECT_EQ(j["type"], "if_else");

    EXPECT_TRUE(j.contains("branches"));
    EXPECT_EQ(j["branches"].size(), 2);
    EXPECT_EQ(j["branches"][0]["condition"], "True");
    EXPECT_EQ(j["branches"][1]["condition"], "False");
    EXPECT_EQ(j["branches"][0]["root"]["type"], "sequence");
    EXPECT_EQ(j["branches"][0]["root"]["children"].size(), 1);
    EXPECT_EQ(j["branches"][0]["root"]["children"][0]["type"], "block");
    EXPECT_EQ(j["branches"][1]["root"]["type"], "sequence");
    EXPECT_EQ(j["branches"][1]["root"]["children"].size(), 1);
    EXPECT_EQ(j["branches"][1]["root"]["children"][0]["type"], "block");
}

TEST(JSONSerializerTest, WhileToJSON_break) {
    // Create a sample While node
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    auto& scope = builder.add_while(root);
    auto& body = builder.add_block(scope.root());
    auto& break_state = builder.add_break(scope.root());

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the While node to JSON
    nlohmann::json j;
    serializer.while_node_to_json(j, scope);

    // Check if the JSON contains the expected keys
    EXPECT_TRUE(j.contains("type"));
    EXPECT_EQ(j["type"], "while");

    EXPECT_TRUE(j.contains("root"));
    EXPECT_EQ(j["root"]["type"], "sequence");
    EXPECT_EQ(j["root"]["children"].size(), 2);
    EXPECT_EQ(j["root"]["children"][0]["type"], "block");
    EXPECT_EQ(j["root"]["children"][1]["type"], "break");
}

TEST(JSONSerializerTest, WhileToJSON_continue) {
    // Create a sample While node
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    auto& scope = builder.add_while(root);
    auto& body = builder.add_block(scope.root());
    auto& continue_state = builder.add_continue(scope.root());

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the While node to JSON
    nlohmann::json j;
    serializer.while_node_to_json(j, scope);

    // Check if the JSON contains the expected keys
    EXPECT_TRUE(j.contains("type"));
    EXPECT_EQ(j["type"], "while");

    EXPECT_TRUE(j.contains("root"));
    EXPECT_EQ(j["root"]["type"], "sequence");
    EXPECT_EQ(j["root"]["children"].size(), 2);
    EXPECT_EQ(j["root"]["children"][0]["type"], "block");
    EXPECT_EQ(j["root"]["children"][1]["type"], "continue");
}

TEST(JSONSerializerTest, ReturnToJSON) {
    // Create a sample Return node
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    auto& scope = builder.add_return(root);

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the Return node to JSON
    nlohmann::json j;
    serializer.return_node_to_json(j, scope);

    // Check if the JSON contains the expected keys
    EXPECT_TRUE(j.contains("type"));
    EXPECT_EQ(j["type"], "return");
}

TEST(JSONSerializerTest, SequenceToJSON) {
    // Create a sample Sequence node
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    auto& scope = builder.add_sequence(root);
    auto& block1 = builder.add_block(scope);
    auto& block2 = builder.add_block(scope, {{symbolic::symbol("i"), symbolic::integer(0)}});

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the Sequence node to JSON
    nlohmann::json j;
    serializer.sequence_to_json(j, scope);

    // Check if the JSON contains the expected keys
    EXPECT_TRUE(j.contains("type"));
    EXPECT_EQ(j["type"], "sequence");
    EXPECT_TRUE(j.contains("children"));
    EXPECT_EQ(j["children"].size(), 2);
    EXPECT_EQ(j["children"][0]["type"], "block");
    EXPECT_EQ(j["children"][1]["type"], "block");

    EXPECT_TRUE(j.contains("transitions"));
    EXPECT_TRUE(j["transitions"].is_array());
    EXPECT_EQ(j["transitions"].size(), 2);
    EXPECT_TRUE(j["transitions"][0].contains("type"));
    EXPECT_EQ(j["transitions"][0]["type"], "transition");
    EXPECT_TRUE(j["transitions"][0].contains("assignments"));
    EXPECT_TRUE(j["transitions"][0]["assignments"].is_array());
    EXPECT_EQ(j["transitions"][0]["assignments"].size(), 0);

    EXPECT_TRUE(j["transitions"][1].contains("type"));
    EXPECT_EQ(j["transitions"][1]["type"], "transition");
    EXPECT_TRUE(j["transitions"][1].contains("assignments"));
    EXPECT_TRUE(j["transitions"][1]["assignments"].is_array());
    EXPECT_EQ(j["transitions"][1]["assignments"].size(), 1);
    EXPECT_TRUE(j["transitions"][1]["assignments"][0].contains("symbol"));
    EXPECT_EQ(j["transitions"][1]["assignments"][0]["symbol"], "i");
    EXPECT_TRUE(j["transitions"][1]["assignments"][0].contains("expression"));
    // EXPECT_TRUE(j["transitions"][1]["assignments"][0]["expression"], "0");
}

TEST(JSONSerializerTest, MapToJSON) {
    // Create a sample Map node
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);

    auto& map = builder.add_map(root, symbolic::symbol("i"), symbolic::integer(10),
                                structured_control_flow::ScheduleType_Sequential);
    auto& body = builder.add_block(map.root());

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the Map node to JSON
    nlohmann::json j;
    serializer.map_to_json(j, map);

    // Check if the JSON contains the expected keys
    EXPECT_TRUE(j.contains("type"));
    EXPECT_EQ(j["type"], "map");
    EXPECT_TRUE(j.contains("indvar"));
    EXPECT_EQ(j["indvar"], "i");
    EXPECT_TRUE(j.contains("num_iterations"));
    EXPECT_EQ(j["num_iterations"], "10");
    EXPECT_TRUE(j.contains("root"));
    EXPECT_EQ(j["root"]["type"], "sequence");
    EXPECT_EQ(j["root"]["children"].size(), 1);
}

TEST(JSONSerializerTest, SerializeDeserializeDataType_Scalar) {
    // Create a sample Scalar data type
    types::Scalar scalar_type(types::StorageType_CPU_Stack, 0, "initializer",
                              types::PrimitiveType::Int32);
    nlohmann::json j;

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the data type to JSON
    serializer.type_to_json(j, scalar_type);

    // Deserialize the JSON back into a Scalar data type
    auto deserialized_scalar_type = serializer.json_to_type(j);
    EXPECT_TRUE(deserialized_scalar_type != nullptr);
    EXPECT_TRUE(dynamic_cast<types::Scalar*>(deserialized_scalar_type.get()) != nullptr);
    auto scalar_ptr = dynamic_cast<types::Scalar*>(deserialized_scalar_type.get());
    EXPECT_TRUE(scalar_ptr != nullptr);

    // Check if the deserialized data type matches the original data type
    EXPECT_EQ(scalar_ptr->primitive_type(), scalar_type.primitive_type());
    EXPECT_EQ(scalar_ptr->storage_type(), scalar_type.storage_type());
    EXPECT_EQ(scalar_ptr->alignment(), scalar_type.alignment());
    EXPECT_EQ(scalar_ptr->initializer(), scalar_type.initializer());
}

TEST(JSONSerializerTest, SerializeDeserializeDataType_Pointer) {
    // Create a sample Pointer data type
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer pointer_type(types::StorageType_CPU_Stack, 0, "initializer", base_desc);
    nlohmann::json j;

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the data type to JSON
    serializer.type_to_json(j, pointer_type);

    // Deserialize the JSON back into a Pointer data type
    auto deserialized_pointer_type = serializer.json_to_type(j);

    // Check if the deserialized data type is of type Pointer
    EXPECT_TRUE(dynamic_cast<types::Pointer*>(deserialized_pointer_type.get()) != nullptr);
    auto pointer_ptr = dynamic_cast<types::Pointer*>(deserialized_pointer_type.get());

    // Check if the deserialized data type matches the original data type
    EXPECT_EQ(pointer_ptr->storage_type().value(), pointer_type.storage_type().value());
    EXPECT_EQ(pointer_ptr->alignment(), pointer_type.alignment());
    EXPECT_EQ(pointer_ptr->initializer(), pointer_type.initializer());
    EXPECT_EQ(pointer_ptr->pointee_type().primitive_type(), base_desc.primitive_type());

    // Check if the deserialized pointee data type matches the original pointee data type
    auto deserialized_pointee_type = &pointer_ptr->pointee_type();
    EXPECT_TRUE(dynamic_cast<const types::Scalar*>(deserialized_pointee_type) != nullptr);
    auto deserialized_base_desc = dynamic_cast<const types::Scalar*>(deserialized_pointee_type);
    EXPECT_TRUE(deserialized_base_desc != nullptr);
    EXPECT_EQ(deserialized_base_desc->primitive_type(), base_desc.primitive_type());
    EXPECT_EQ(deserialized_base_desc->storage_type().value(), base_desc.storage_type().value());
    EXPECT_EQ(deserialized_base_desc->alignment(), base_desc.alignment());
    EXPECT_EQ(deserialized_base_desc->initializer(), base_desc.initializer());
}

TEST(JSONSerializerTest, SerializeDeserializeDataType_Structure) {
    // Create a sample Structure data type
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Structure structure_type(types::StorageType_CPU_Stack, 0, "initializer", "MyStruct");
    nlohmann::json j;

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto sdfg = builder.move();

    sdfg::serializer::JSONSerializer serializer;
    // Serialize the data type to JSON
    serializer.type_to_json(j, structure_type);
    // Deserialize the JSON back into a Structure data type
    auto deserialized_structure_type = serializer.json_to_type(j);

    // Check if the deserialized data type is of type Structure
    EXPECT_TRUE(dynamic_cast<types::Structure*>(deserialized_structure_type.get()) != nullptr);
    auto structure_ptr = dynamic_cast<types::Structure*>(deserialized_structure_type.get());
    EXPECT_TRUE(structure_ptr != nullptr);
    // Check if the deserialized data type matches the original data type
    EXPECT_EQ(structure_ptr->name(), structure_type.name());
    EXPECT_EQ(structure_ptr->storage_type().value(), structure_type.storage_type().value());
    EXPECT_EQ(structure_ptr->alignment(), structure_type.alignment());
    EXPECT_EQ(structure_ptr->initializer(), structure_type.initializer());
}

TEST(JSONSerializerTest, SerializeDeserializeDataType_Array) {
    // Create a sample Array data type
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_type(types::StorageType_CPU_Stack, 0, "initializer", base_desc,
                            {symbolic::symbol("N")});
    nlohmann::json j;

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the data type to JSON
    serializer.type_to_json(j, array_type);

    // Deserialize the JSON back into an Array data type
    auto deserialized_array_type = serializer.json_to_type(j);

    // Check if the deserialized data type is of type Array
    EXPECT_TRUE(dynamic_cast<types::Array*>(deserialized_array_type.get()) != nullptr);
    auto array_ptr = dynamic_cast<types::Array*>(deserialized_array_type.get());
    EXPECT_TRUE(array_ptr != nullptr);

    // Check if the deserialized data type matches the original data type
    EXPECT_EQ(array_ptr->storage_type().value(), array_type.storage_type().value());
    EXPECT_EQ(array_ptr->alignment(), array_type.alignment());
    EXPECT_EQ(array_ptr->initializer(), array_type.initializer());
    EXPECT_EQ(array_ptr->num_elements()->__str__(), "N");

    // Check if the deserialized element type matches the original element type
    auto deserialized_element_type = &array_ptr->element_type();
    EXPECT_TRUE(dynamic_cast<const types::Scalar*>(deserialized_element_type) != nullptr);
    auto deserialized_base_desc = dynamic_cast<const types::Scalar*>(deserialized_element_type);
    EXPECT_TRUE(deserialized_base_desc != nullptr);
    EXPECT_EQ(deserialized_base_desc->primitive_type(), base_desc.primitive_type());
    EXPECT_EQ(deserialized_base_desc->storage_type().value(), base_desc.storage_type().value());
    EXPECT_EQ(deserialized_base_desc->alignment(), base_desc.alignment());
    EXPECT_EQ(deserialized_base_desc->initializer(), base_desc.initializer());
}

TEST(JSONSerializerTest, SerializeDeserializeDataType_Function) {
    // Create a sample Scalar data type
    types::Scalar scalar_type(types::StorageType_CPU_Stack, 0, "initializer",
                              types::PrimitiveType::Int32);
    types::Function function_type(scalar_type, false);
    function_type.add_param(scalar_type);
    function_type.add_param(scalar_type);
    nlohmann::json j;

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the data type to JSON
    serializer.type_to_json(j, function_type);

    // Deserialize the JSON back into a Scalar data type
    auto deserialized_function_type = serializer.json_to_type(j);
    EXPECT_TRUE(deserialized_function_type != nullptr);
    EXPECT_TRUE(dynamic_cast<types::Function*>(deserialized_function_type.get()) != nullptr);
    auto function_ptr = dynamic_cast<types::Function*>(deserialized_function_type.get());
    EXPECT_TRUE(function_ptr != nullptr);

    // Check if the deserialized data type matches the original data type
    EXPECT_EQ(function_ptr->primitive_type(), function_type.primitive_type());
    EXPECT_EQ(function_ptr->storage_type().value(), function_type.storage_type().value());
    EXPECT_EQ(function_ptr->alignment(), function_type.alignment());
    EXPECT_EQ(function_ptr->initializer(), function_type.initializer());
    EXPECT_EQ(function_ptr->is_var_arg(), function_type.is_var_arg());
    EXPECT_EQ(function_ptr->num_params(), function_type.num_params());
    EXPECT_EQ(function_ptr->num_params(), 2);
    EXPECT_EQ(function_ptr->param_type(symbolic::integer(0)).primitive_type(),
              scalar_type.primitive_type());
    EXPECT_EQ(function_ptr->param_type(symbolic::integer(1)).primitive_type(),
              scalar_type.primitive_type());
}

TEST(JSONSerializerTest, SerializeDeserializeDataType_StructureDefinition) {
    // Create a sample StructureDefinition object
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_type(base_desc, {symbolic::symbol("N")});
    types::StructureDefinition structure_definition("MyStruct", true);
    structure_definition.add_member(base_desc);
    structure_definition.add_member(array_type);
    nlohmann::json j;

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the StructureDefinition to JSON
    serializer.structure_definition_to_json(j, structure_definition);

    // define sdfg builder for deserialization
    sdfg::builder::StructuredSDFGBuilder builder_deserialize("test_sdfg", FunctionType_CPU);

    // Deserialize the JSON back into a StructureDefinition object
    serializer.json_to_structure_definition(j, builder_deserialize);

    auto des_sdfg = builder_deserialize.move();

    EXPECT_EQ(des_sdfg->structures().size(), 1);
    auto& deserialized_structure_definition = des_sdfg->structures().begin().base()->second;

    // Check if the deserialized StructureDefinition matches the original StructureDefinition
    EXPECT_EQ(deserialized_structure_definition->name(), structure_definition.name());
    EXPECT_EQ(deserialized_structure_definition->num_members(), structure_definition.num_members());
    EXPECT_EQ(deserialized_structure_definition->is_packed(), structure_definition.is_packed());

    auto& des_member_0 = deserialized_structure_definition->member_type(symbolic::integer(0));
    auto& member_0 = structure_definition.member_type(symbolic::integer(0));
    EXPECT_TRUE(dynamic_cast<const types::Scalar*>(&des_member_0) != nullptr);
    EXPECT_TRUE(dynamic_cast<const types::Scalar*>(&member_0) != nullptr);
    EXPECT_EQ(des_member_0.primitive_type(), member_0.primitive_type());
    EXPECT_EQ(des_member_0.storage_type().value(), member_0.storage_type().value());
    EXPECT_EQ(des_member_0.initializer(), member_0.initializer());
    EXPECT_EQ(des_member_0.alignment(), member_0.alignment());

    auto& des_member_1 = deserialized_structure_definition->member_type(symbolic::integer(1));
    auto& member_1 = structure_definition.member_type(symbolic::integer(1));
    EXPECT_TRUE(dynamic_cast<const types::Array*>(&des_member_1) != nullptr);
    auto& des_member_1_arr = dynamic_cast<const types::Array&>(des_member_1);
    EXPECT_TRUE(dynamic_cast<const types::Array*>(&member_1) != nullptr);
    auto& member_1_arr = dynamic_cast<const types::Array&>(member_1);

    EXPECT_EQ(des_member_1_arr.element_type().primitive_type(),
              member_1_arr.element_type().primitive_type());
    EXPECT_EQ(des_member_1_arr.storage_type().value(), member_1_arr.storage_type().value());
    EXPECT_EQ(des_member_1_arr.initializer(), member_1_arr.initializer());
    EXPECT_EQ(des_member_1_arr.alignment(), member_1_arr.alignment());
    EXPECT_TRUE(sdfg::symbolic::eq(des_member_1_arr.num_elements(), member_1_arr.num_elements()));
}

TEST(JSONSerializerTest, SerializeDeserialize_DataflowGraph) {
    // Create a sample StructuredSDFG object
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer pointer_type(base_desc);

    builder.add_container("A", pointer_type);
    builder.add_container("C", base_desc);

    auto& block = builder.add_block(root);
    auto& access_in = builder.add_access(block, "A");
    auto& access_in2 = builder.add_access(block, "C");
    auto& access_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::add, {"_out", base_desc},
                                        {{"_in1", base_desc}, {"_in2", base_desc}});
    auto& memlet_in =
        builder.add_memlet(block, access_in, "void", tasklet, "_in1", {{symbolic::symbol("i")}});
    auto& memlet_in2 = builder.add_memlet(block, access_in2, "void", tasklet, "_in2", {});
    auto& memlet_out = builder.add_memlet(block, tasklet, "_out", access_out, "void", {});

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;
    nlohmann::json j;
    // Serialize the DataflowGraph to JSON
    auto& block_new = dynamic_cast<sdfg::structured_control_flow::Block&>(sdfg->root().at(0).first);

    serializer.dataflow_to_json(j, block_new.dataflow());

    // Deserialize the JSON back into a DataflowGraph object
    auto des_builder = sdfg::builder::StructuredSDFGBuilder("test_sdfg", FunctionType_CPU);
    auto& block2 = des_builder.add_block(des_builder.subject().root());

    des_builder.add_container("A", pointer_type);
    des_builder.add_container("C", base_desc);

    serializer.json_to_dataflow(j, des_builder, block2);
    auto des_sdfg = des_builder.move();

    auto& des_block_new =
        dynamic_cast<sdfg::structured_control_flow::Block&>(des_sdfg->root().at(0).first);

    auto& des_dataflow = des_block_new.dataflow();
    auto& data_flow = block_new.dataflow();
    EXPECT_EQ(des_dataflow.nodes().size(), data_flow.nodes().size());
    EXPECT_EQ(des_dataflow.edges().size(), data_flow.edges().size());

    bool foundA = false;
    int foundC = 0;
    bool found_tasklet = false;

    // Check if the deserialized DataflowGraph matches the original DataflowGraph
    for (const auto& node : des_dataflow.nodes()) {
        if (auto access_node = dynamic_cast<const sdfg::data_flow::AccessNode*>(&node)) {
            if (access_node->data() == "A") {
                foundA = true;
                auto& type = des_sdfg->type(access_node->data());
                EXPECT_TRUE(dynamic_cast<const types::Pointer*>(&type) != nullptr);
                auto& type_ptr = dynamic_cast<const types::Pointer&>(type);
                EXPECT_EQ(type_ptr.primitive_type(), base_desc.primitive_type());
                EXPECT_EQ(type_ptr.storage_type().value(), base_desc.storage_type().value());
                EXPECT_EQ(type_ptr.initializer(), base_desc.initializer());
                EXPECT_EQ(type_ptr.alignment(), base_desc.alignment());
                EXPECT_EQ(type_ptr.pointee_type().primitive_type(), base_desc.primitive_type());
            } else if (access_node->data() == "C") {
                foundC++;
                auto& type = des_sdfg->type(access_node->data());
                EXPECT_TRUE(dynamic_cast<const types::Scalar*>(&type) != nullptr);
                auto& type_ptr = dynamic_cast<const types::Scalar&>(type);
                EXPECT_EQ(type_ptr.primitive_type(), base_desc.primitive_type());
                EXPECT_EQ(type_ptr.storage_type().value(), base_desc.storage_type().value());
                EXPECT_EQ(type_ptr.initializer(), base_desc.initializer());
                EXPECT_EQ(type_ptr.alignment(), base_desc.alignment());
            }

        } else if (auto tasklet_node = dynamic_cast<const sdfg::data_flow::Tasklet*>(&node)) {
            EXPECT_EQ(tasklet_node->code(), data_flow::TaskletCode::add);
            EXPECT_EQ(tasklet_node->inputs().size(), 2);
            EXPECT_EQ(tasklet_node->output().first, "_out");
            EXPECT_EQ(tasklet_node->inputs().at(0).first, "_in1");
            EXPECT_EQ(tasklet_node->inputs().at(1).first, "_in2");
            EXPECT_EQ(tasklet_node->inputs().at(0).second.primitive_type(),
                      types::PrimitiveType::Float);
            EXPECT_EQ(tasklet_node->inputs().at(1).second.primitive_type(),
                      types::PrimitiveType::Float);
            EXPECT_EQ(tasklet_node->output().second.primitive_type(), types::PrimitiveType::Float);
            found_tasklet = true;
        }
    }

    EXPECT_TRUE(foundA);
    EXPECT_EQ(foundC, 2);
    EXPECT_TRUE(found_tasklet);
    bool found_memlet_in = false;
    bool found_memlet_in2 = false;
    bool found_memlet_out = false;
    // Check if the deserialized Memlets match the original Memlets
    for (const auto& edge : des_dataflow.edges()) {
        if (auto memlet = dynamic_cast<const sdfg::data_flow::Memlet*>(&edge)) {
            if (memlet->dst_conn() == "_in1") {
                found_memlet_in = true;
                auto& src = memlet->src();
                EXPECT_TRUE(dynamic_cast<const sdfg::data_flow::AccessNode*>(&src) != nullptr);
                auto& src_ptr = dynamic_cast<const sdfg::data_flow::AccessNode&>(src);
                EXPECT_EQ(src_ptr.data(), "A");

                auto& subset = memlet->subset();
                EXPECT_EQ(subset.size(), 1);
                EXPECT_TRUE(symbolic::eq(subset[0], symbolic::symbol("i")));
            } else if (memlet->dst_conn() == "_in2") {
                found_memlet_in2 = true;
                auto& src = memlet->src();
                EXPECT_TRUE(dynamic_cast<const sdfg::data_flow::AccessNode*>(&src) != nullptr);
                auto& src_ptr = dynamic_cast<const sdfg::data_flow::AccessNode&>(src);
                EXPECT_EQ(src_ptr.data(), "C");
            } else if (memlet->src_conn() == "_out") {
                found_memlet_out = true;
                auto& dst = memlet->dst();
                EXPECT_TRUE(dynamic_cast<const sdfg::data_flow::AccessNode*>(&dst) != nullptr);
                auto& dst_ptr = dynamic_cast<const sdfg::data_flow::AccessNode&>(dst);
                EXPECT_EQ(dst_ptr.data(), "C");
            }
        }
    }

    EXPECT_TRUE(found_memlet_in);
    EXPECT_TRUE(found_memlet_in2);
    EXPECT_TRUE(found_memlet_out);
}

TEST(JSONSerializerTest, SerializeDeserializeBlock_DataflowGraph) {
    // Create a sample StructuredSDFG object
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer pointer_type(base_desc);

    builder.add_container("A", pointer_type);
    builder.add_container("C", base_desc);

    auto& block = builder.add_block(root);
    auto& access_in = builder.add_access(block, "A");
    auto& access_in2 = builder.add_access(block, "C");
    auto& access_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::add, {"_out", base_desc},
                                        {{"_in1", base_desc}, {"_in2", base_desc}});
    auto& memlet_in =
        builder.add_memlet(block, access_in, "void", tasklet, "_in1", {{symbolic::symbol("i")}});
    auto& memlet_in2 = builder.add_memlet(block, access_in2, "void", tasklet, "_in2", {});
    auto& memlet_out = builder.add_memlet(block, tasklet, "_out", access_out, "void", {});

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;
    nlohmann::json j;
    // Serialize the DataflowGraph to JSON
    auto& block_new = dynamic_cast<sdfg::structured_control_flow::Block&>(sdfg->root().at(0).first);

    serializer.block_to_json(j, block_new);

    // Deserialize the JSON back into a DataflowGraph object
    auto des_builder = sdfg::builder::StructuredSDFGBuilder("test_sdfg", FunctionType_CPU);

    des_builder.add_container("A", pointer_type);
    des_builder.add_container("C", base_desc);

    control_flow::Assignments assignments;

    serializer.json_to_block_node(j, des_builder, des_builder.subject().root(), assignments);
    auto des_sdfg = des_builder.move();

    auto& des_block_new =
        dynamic_cast<sdfg::structured_control_flow::Block&>(des_sdfg->root().at(0).first);

    auto& des_dataflow = des_block_new.dataflow();
    auto& data_flow = block_new.dataflow();
    EXPECT_EQ(des_dataflow.nodes().size(), data_flow.nodes().size());
    EXPECT_EQ(des_dataflow.edges().size(), data_flow.edges().size());

    bool foundA = false;
    int foundC = 0;
    bool found_tasklet = false;

    // Check if the deserialized DataflowGraph matches the original DataflowGraph
    for (const auto& node : des_dataflow.nodes()) {
        if (auto access_node = dynamic_cast<const sdfg::data_flow::AccessNode*>(&node)) {
            if (access_node->data() == "A") {
                foundA = true;
                auto& type = des_sdfg->type(access_node->data());
                EXPECT_TRUE(dynamic_cast<const types::Pointer*>(&type) != nullptr);
                auto& type_ptr = dynamic_cast<const types::Pointer&>(type);
                EXPECT_EQ(type_ptr.primitive_type(), base_desc.primitive_type());
                EXPECT_EQ(type_ptr.storage_type().value(), base_desc.storage_type().value());
                EXPECT_EQ(type_ptr.initializer(), base_desc.initializer());
                EXPECT_EQ(type_ptr.alignment(), base_desc.alignment());
                EXPECT_EQ(type_ptr.pointee_type().primitive_type(), base_desc.primitive_type());
            } else if (access_node->data() == "C") {
                foundC++;
                auto& type = des_sdfg->type(access_node->data());
                EXPECT_TRUE(dynamic_cast<const types::Scalar*>(&type) != nullptr);
                auto& type_ptr = dynamic_cast<const types::Scalar&>(type);
                EXPECT_EQ(type_ptr.primitive_type(), base_desc.primitive_type());
                EXPECT_EQ(type_ptr.storage_type().value(), base_desc.storage_type().value());
                EXPECT_EQ(type_ptr.initializer(), base_desc.initializer());
                EXPECT_EQ(type_ptr.alignment(), base_desc.alignment());
            }

        } else if (auto tasklet_node = dynamic_cast<const sdfg::data_flow::Tasklet*>(&node)) {
            EXPECT_EQ(tasklet_node->code(), data_flow::TaskletCode::add);
            EXPECT_EQ(tasklet_node->inputs().size(), 2);
            EXPECT_EQ(tasklet_node->output().first, "_out");
            EXPECT_EQ(tasklet_node->inputs().at(0).first, "_in1");
            EXPECT_EQ(tasklet_node->inputs().at(1).first, "_in2");
            EXPECT_EQ(tasklet_node->inputs().at(0).second.primitive_type(),
                      types::PrimitiveType::Float);
            EXPECT_EQ(tasklet_node->inputs().at(1).second.primitive_type(),
                      types::PrimitiveType::Float);
            EXPECT_EQ(tasklet_node->output().second.primitive_type(), types::PrimitiveType::Float);
            found_tasklet = true;
        }
    }

    EXPECT_TRUE(foundA);
    EXPECT_EQ(foundC, 2);
    EXPECT_TRUE(found_tasklet);
    bool found_memlet_in = false;
    bool found_memlet_in2 = false;
    bool found_memlet_out = false;
    // Check if the deserialized Memlets match the original Memlets
    for (const auto& edge : des_dataflow.edges()) {
        if (auto memlet = dynamic_cast<const sdfg::data_flow::Memlet*>(&edge)) {
            if (memlet->dst_conn() == "_in1") {
                found_memlet_in = true;
                auto& src = memlet->src();
                EXPECT_TRUE(dynamic_cast<const sdfg::data_flow::AccessNode*>(&src) != nullptr);
                auto& src_ptr = dynamic_cast<const sdfg::data_flow::AccessNode&>(src);
                EXPECT_EQ(src_ptr.data(), "A");

                auto& subset = memlet->subset();
                EXPECT_EQ(subset.size(), 1);
                EXPECT_TRUE(symbolic::eq(subset[0], symbolic::symbol("i")));
            } else if (memlet->dst_conn() == "_in2") {
                found_memlet_in2 = true;
                auto& src = memlet->src();
                EXPECT_TRUE(dynamic_cast<const sdfg::data_flow::AccessNode*>(&src) != nullptr);
                auto& src_ptr = dynamic_cast<const sdfg::data_flow::AccessNode&>(src);
                EXPECT_EQ(src_ptr.data(), "C");
            } else if (memlet->src_conn() == "_out") {
                found_memlet_out = true;
                auto& dst = memlet->dst();
                EXPECT_TRUE(dynamic_cast<const sdfg::data_flow::AccessNode*>(&dst) != nullptr);
                auto& dst_ptr = dynamic_cast<const sdfg::data_flow::AccessNode&>(dst);
                EXPECT_EQ(dst_ptr.data(), "C");
            }
        }
    }

    EXPECT_TRUE(found_memlet_in);
    EXPECT_TRUE(found_memlet_in2);
    EXPECT_TRUE(found_memlet_out);
}

TEST(JSONSerializerTest, SerializeDeserializeSequence_DataflowGraph) {
    // Create a sample StructuredSDFG object
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer pointer_type(base_desc);

    builder.add_container("A", pointer_type);
    builder.add_container("C", base_desc);

    auto& block = builder.add_block(root);
    auto& access_in = builder.add_access(block, "A");
    auto& access_in2 = builder.add_access(block, "C");
    auto& access_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::add, {"_out", base_desc},
                                        {{"_in1", base_desc}, {"_in2", base_desc}});
    auto& memlet_in =
        builder.add_memlet(block, access_in, "void", tasklet, "_in1", {{symbolic::symbol("i")}});
    auto& memlet_in2 = builder.add_memlet(block, access_in2, "void", tasklet, "_in2", {});
    auto& memlet_out = builder.add_memlet(block, tasklet, "_out", access_out, "void", {});

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;
    nlohmann::json j;
    // Serialize the DataflowGraph to JSON
    auto& block_new = dynamic_cast<sdfg::structured_control_flow::Block&>(sdfg->root().at(0).first);

    serializer.sequence_to_json(j, sdfg->root());

    // Deserialize the JSON back into a DataflowGraph object
    auto des_builder = sdfg::builder::StructuredSDFGBuilder("test_sdfg", FunctionType_CPU);

    des_builder.add_container("A", pointer_type);
    des_builder.add_container("C", base_desc);

    serializer.json_to_sequence(j, des_builder, des_builder.subject().root());
    auto des_sdfg = des_builder.move();

    EXPECT_EQ(des_sdfg->name(), sdfg->name());
    EXPECT_EQ(des_sdfg->containers().size(), 2);
    EXPECT_EQ(des_sdfg->root().size(), 1);

    EXPECT_TRUE(dynamic_cast<sdfg::structured_control_flow::Block*>(
                    &des_sdfg->root().at(0).first) != nullptr);

    auto& des_block_new =
        dynamic_cast<sdfg::structured_control_flow::Block&>(des_sdfg->root().at(0).first);

    auto& des_dataflow = des_block_new.dataflow();
    auto& data_flow = block_new.dataflow();
    EXPECT_EQ(des_dataflow.nodes().size(), data_flow.nodes().size());
    EXPECT_EQ(des_dataflow.edges().size(), data_flow.edges().size());

    bool foundA = false;
    int foundC = 0;
    bool found_tasklet = false;

    // Check if the deserialized DataflowGraph matches the original DataflowGraph
    for (const auto& node : des_dataflow.nodes()) {
        if (auto access_node = dynamic_cast<const sdfg::data_flow::AccessNode*>(&node)) {
            if (access_node->data() == "A") {
                foundA = true;
                auto& type = des_sdfg->type(access_node->data());
                EXPECT_TRUE(dynamic_cast<const types::Pointer*>(&type) != nullptr);
                auto& type_ptr = dynamic_cast<const types::Pointer&>(type);
                EXPECT_EQ(type_ptr.primitive_type(), base_desc.primitive_type());
                EXPECT_EQ(type_ptr.storage_type().value(), base_desc.storage_type().value());
                EXPECT_EQ(type_ptr.initializer(), base_desc.initializer());
                EXPECT_EQ(type_ptr.alignment(), base_desc.alignment());
                EXPECT_EQ(type_ptr.pointee_type().primitive_type(), base_desc.primitive_type());
            } else if (access_node->data() == "C") {
                foundC++;
                auto& type = des_sdfg->type(access_node->data());
                EXPECT_TRUE(dynamic_cast<const types::Scalar*>(&type) != nullptr);
                auto& type_ptr = dynamic_cast<const types::Scalar&>(type);
                EXPECT_EQ(type_ptr.primitive_type(), base_desc.primitive_type());
                EXPECT_EQ(type_ptr.storage_type().value(), base_desc.storage_type().value());
                EXPECT_EQ(type_ptr.initializer(), base_desc.initializer());
                EXPECT_EQ(type_ptr.alignment(), base_desc.alignment());
            }

        } else if (auto tasklet_node = dynamic_cast<const sdfg::data_flow::Tasklet*>(&node)) {
            EXPECT_EQ(tasklet_node->code(), data_flow::TaskletCode::add);
            EXPECT_EQ(tasklet_node->inputs().size(), 2);
            EXPECT_EQ(tasklet_node->output().first, "_out");
            EXPECT_EQ(tasklet_node->inputs().at(0).first, "_in1");
            EXPECT_EQ(tasklet_node->inputs().at(1).first, "_in2");
            EXPECT_EQ(tasklet_node->inputs().at(0).second.primitive_type(),
                      types::PrimitiveType::Float);
            EXPECT_EQ(tasklet_node->inputs().at(1).second.primitive_type(),
                      types::PrimitiveType::Float);
            EXPECT_EQ(tasklet_node->output().second.primitive_type(), types::PrimitiveType::Float);
            found_tasklet = true;
        }
    }

    EXPECT_TRUE(foundA);
    EXPECT_EQ(foundC, 2);
    EXPECT_TRUE(found_tasklet);
    bool found_memlet_in = false;
    bool found_memlet_in2 = false;
    bool found_memlet_out = false;
    // Check if the deserialized Memlets match the original Memlets
    for (const auto& edge : des_dataflow.edges()) {
        if (auto memlet = dynamic_cast<const sdfg::data_flow::Memlet*>(&edge)) {
            if (memlet->dst_conn() == "_in1") {
                found_memlet_in = true;
                auto& src = memlet->src();
                EXPECT_TRUE(dynamic_cast<const sdfg::data_flow::AccessNode*>(&src) != nullptr);
                auto& src_ptr = dynamic_cast<const sdfg::data_flow::AccessNode&>(src);
                EXPECT_EQ(src_ptr.data(), "A");

                auto& subset = memlet->subset();
                EXPECT_EQ(subset.size(), 1);
                EXPECT_TRUE(symbolic::eq(subset[0], symbolic::symbol("i")));
            } else if (memlet->dst_conn() == "_in2") {
                found_memlet_in2 = true;
                auto& src = memlet->src();
                EXPECT_TRUE(dynamic_cast<const sdfg::data_flow::AccessNode*>(&src) != nullptr);
                auto& src_ptr = dynamic_cast<const sdfg::data_flow::AccessNode&>(src);
                EXPECT_EQ(src_ptr.data(), "C");
            } else if (memlet->src_conn() == "_out") {
                found_memlet_out = true;
                auto& dst = memlet->dst();
                EXPECT_TRUE(dynamic_cast<const sdfg::data_flow::AccessNode*>(&dst) != nullptr);
                auto& dst_ptr = dynamic_cast<const sdfg::data_flow::AccessNode&>(dst);
                EXPECT_EQ(dst_ptr.data(), "C");
            }
        }
    }

    EXPECT_TRUE(found_memlet_in);
    EXPECT_TRUE(found_memlet_in2);
    EXPECT_TRUE(found_memlet_out);
}

TEST(JSONSerializerTest, SerializeDeserializeSDFG_DataflowGraph) {
    // Create a sample StructuredSDFG object
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer pointer_type(base_desc);

    builder.add_container("A", pointer_type);
    builder.add_container("C", base_desc);

    auto& block = builder.add_block(root);
    auto& access_in = builder.add_access(block, "A");
    auto& access_in2 = builder.add_access(block, "C");
    auto& access_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::add, {"_out", base_desc},
                                        {{"_in1", base_desc}, {"_in2", base_desc}});
    auto& memlet_in =
        builder.add_memlet(block, access_in, "void", tasklet, "_in1", {{symbolic::symbol("i")}});
    auto& memlet_in2 = builder.add_memlet(block, access_in2, "void", tasklet, "_in2", {});
    auto& memlet_out = builder.add_memlet(block, tasklet, "_out", access_out, "void", {});

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;
    nlohmann::json j;
    // Serialize the DataflowGraph to JSON
    auto& block_new = dynamic_cast<sdfg::structured_control_flow::Block&>(sdfg->root().at(0).first);

    j = serializer.serialize(sdfg);

    // Deserialize the JSON back into a DataflowGraph object
    auto des_sdfg = serializer.deserialize(j);

    EXPECT_EQ(des_sdfg->name(), sdfg->name());
    EXPECT_EQ(des_sdfg->containers().size(), 2);
    EXPECT_EQ(des_sdfg->root().size(), 1);

    EXPECT_TRUE(dynamic_cast<sdfg::structured_control_flow::Block*>(
                    &des_sdfg->root().at(0).first) != nullptr);

    auto& des_block_new =
        dynamic_cast<sdfg::structured_control_flow::Block&>(des_sdfg->root().at(0).first);

    auto& des_dataflow = des_block_new.dataflow();
    auto& data_flow = block_new.dataflow();
    EXPECT_EQ(des_dataflow.nodes().size(), data_flow.nodes().size());
    EXPECT_EQ(des_dataflow.edges().size(), data_flow.edges().size());

    bool foundA = false;
    int foundC = 0;
    bool found_tasklet = false;

    // Check if the deserialized DataflowGraph matches the original DataflowGraph
    for (const auto& node : des_dataflow.nodes()) {
        if (auto access_node = dynamic_cast<const sdfg::data_flow::AccessNode*>(&node)) {
            if (access_node->data() == "A") {
                foundA = true;
                auto& type = des_sdfg->type(access_node->data());
                EXPECT_TRUE(dynamic_cast<const types::Pointer*>(&type) != nullptr);
                auto& type_ptr = dynamic_cast<const types::Pointer&>(type);
                EXPECT_EQ(type_ptr.primitive_type(), base_desc.primitive_type());
                EXPECT_EQ(type_ptr.storage_type(), base_desc.storage_type());
                EXPECT_EQ(type_ptr.initializer(), base_desc.initializer());
                EXPECT_EQ(type_ptr.alignment(), base_desc.alignment());
                EXPECT_EQ(type_ptr.pointee_type().primitive_type(), base_desc.primitive_type());
            } else if (access_node->data() == "C") {
                foundC++;
                auto& type = des_sdfg->type(access_node->data());
                EXPECT_TRUE(dynamic_cast<const types::Scalar*>(&type) != nullptr);
                auto& type_ptr = dynamic_cast<const types::Scalar&>(type);
                EXPECT_EQ(type_ptr.primitive_type(), base_desc.primitive_type());
                EXPECT_EQ(type_ptr.storage_type(), base_desc.storage_type());
                EXPECT_EQ(type_ptr.initializer(), base_desc.initializer());
                EXPECT_EQ(type_ptr.alignment(), base_desc.alignment());
            }

        } else if (auto tasklet_node = dynamic_cast<const sdfg::data_flow::Tasklet*>(&node)) {
            EXPECT_EQ(tasklet_node->code(), data_flow::TaskletCode::add);
            EXPECT_EQ(tasklet_node->inputs().size(), 2);
            EXPECT_EQ(tasklet_node->inputs().at(0).first, "_in1");
            EXPECT_EQ(tasklet_node->inputs().at(1).first, "_in2");
            EXPECT_EQ(tasklet_node->output().first, "_out");
            EXPECT_EQ(tasklet_node->inputs().at(0).second.primitive_type(),
                      types::PrimitiveType::Float);
            EXPECT_EQ(tasklet_node->inputs().at(1).second.primitive_type(),
                      types::PrimitiveType::Float);
            EXPECT_EQ(tasklet_node->output().second.primitive_type(), types::PrimitiveType::Float);
            found_tasklet = true;
        }
    }

    EXPECT_TRUE(foundA);
    EXPECT_EQ(foundC, 2);
    EXPECT_TRUE(found_tasklet);
    bool found_memlet_in = false;
    bool found_memlet_in2 = false;
    bool found_memlet_out = false;
    // Check if the deserialized Memlets match the original Memlets
    for (const auto& edge : des_dataflow.edges()) {
        if (auto memlet = dynamic_cast<const sdfg::data_flow::Memlet*>(&edge)) {
            if (memlet->dst_conn() == "_in1") {
                found_memlet_in = true;
                auto& src = memlet->src();
                EXPECT_TRUE(dynamic_cast<const sdfg::data_flow::AccessNode*>(&src) != nullptr);
                auto& src_ptr = dynamic_cast<const sdfg::data_flow::AccessNode&>(src);
                EXPECT_EQ(src_ptr.data(), "A");

                auto& subset = memlet->subset();
                EXPECT_EQ(subset.size(), 1);
                EXPECT_TRUE(symbolic::eq(subset[0], symbolic::symbol("i")));
            } else if (memlet->dst_conn() == "_in2") {
                found_memlet_in2 = true;
                auto& src = memlet->src();
                EXPECT_TRUE(dynamic_cast<const sdfg::data_flow::AccessNode*>(&src) != nullptr);
                auto& src_ptr = dynamic_cast<const sdfg::data_flow::AccessNode&>(src);
                EXPECT_EQ(src_ptr.data(), "C");
            } else if (memlet->src_conn() == "_out") {
                found_memlet_out = true;
                auto& dst = memlet->dst();
                EXPECT_TRUE(dynamic_cast<const sdfg::data_flow::AccessNode*>(&dst) != nullptr);
                auto& dst_ptr = dynamic_cast<const sdfg::data_flow::AccessNode&>(dst);
                EXPECT_EQ(dst_ptr.data(), "C");
            }
        }
    }

    EXPECT_TRUE(found_memlet_in);
    EXPECT_TRUE(found_memlet_in2);
    EXPECT_TRUE(found_memlet_out);
}

TEST(JSONSerializerTest, SerializeDeserialize_forloop) {
    // Create a sample StructuredSDFG object
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Int32);

    builder.add_container("i", base_desc);
    builder.add_container("N", base_desc);

    auto loopvar = symbolic::symbol("i");
    auto bound = symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N"));
    auto update = symbolic::add(symbolic::symbol("i"), symbolic::integer(1));
    auto init = symbolic::integer(0);

    auto& for_loop = builder.add_for(root, loopvar, bound, init, update);

    auto& block = builder.add_block(for_loop.root());

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;
    nlohmann::json j;

    // Serialize the DataflowGraph to JSON

    serializer.for_to_json(j, for_loop);

    // Deserialize the JSON back into a DataflowGraph object
    auto des_builder = sdfg::builder::StructuredSDFGBuilder("test_sdfg", FunctionType_CPU);

    des_builder.add_container("i", base_desc);
    des_builder.add_container("N", base_desc);

    control_flow::Assignments assignments;

    serializer.json_to_for_node(j, des_builder, des_builder.subject().root(), assignments);
    auto des_sdfg = des_builder.move();

    EXPECT_EQ(des_sdfg->name(), sdfg->name());
    EXPECT_EQ(des_sdfg->containers().size(), 2);
    EXPECT_EQ(des_sdfg->root().size(), 1);
    EXPECT_TRUE(dynamic_cast<sdfg::structured_control_flow::For*>(&des_sdfg->root().at(0).first) !=
                nullptr);
    auto& des_for_loop =
        dynamic_cast<sdfg::structured_control_flow::For&>(des_sdfg->root().at(0).first);
    EXPECT_TRUE(symbolic::eq(des_for_loop.indvar(), loopvar));
    EXPECT_TRUE(symbolic::eq(des_for_loop.condition(), bound));
    EXPECT_TRUE(symbolic::eq(des_for_loop.update(), update));
    EXPECT_TRUE(symbolic::eq(des_for_loop.init(), init));

    EXPECT_TRUE(dynamic_cast<sdfg::structured_control_flow::Block*>(
                    &des_for_loop.root().at(0).first) != nullptr);
    auto& des_block_new =
        dynamic_cast<sdfg::structured_control_flow::Block&>(des_for_loop.root().at(0).first);
}

TEST(JSONSerializerTest, SerializeDeserialize_ifelse) {
    // Create a sample IfElse node
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    auto& if_else = builder.add_if_else(root);
    auto& true_case = builder.add_case(if_else, symbolic::__true__());
    auto& false_case = builder.add_case(if_else, symbolic::__false__());
    auto& true_block = builder.add_block(true_case);
    auto& false_block = builder.add_block(false_case);

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the IfElse node to JSON
    nlohmann::json j;
    serializer.if_else_to_json(j, if_else);

    // Deserialize the JSON back into an IfElse node
    auto des_builder = sdfg::builder::StructuredSDFGBuilder("test_sdfg", FunctionType_CPU);
    des_builder.add_container("i", sym_desc);
    control_flow::Assignments assignments;
    serializer.json_to_if_else_node(j, des_builder, des_builder.subject().root(), assignments);
    auto des_sdfg = des_builder.move();

    EXPECT_EQ(des_sdfg->name(), sdfg->name());
    EXPECT_EQ(des_sdfg->containers().size(), 1);
    EXPECT_EQ(des_sdfg->root().size(), 1);
    EXPECT_TRUE(dynamic_cast<sdfg::structured_control_flow::IfElse*>(
                    &des_sdfg->root().at(0).first) != nullptr);
    auto& des_if_else =
        dynamic_cast<sdfg::structured_control_flow::IfElse&>(des_sdfg->root().at(0).first);

    EXPECT_EQ(des_if_else.size(), 2);

    EXPECT_TRUE(symbolic::eq(des_if_else.at(0).second, symbolic::__true__()));
    EXPECT_TRUE(symbolic::eq(des_if_else.at(1).second, symbolic::__false__()));
    EXPECT_TRUE(dynamic_cast<sdfg::structured_control_flow::Sequence*>(&des_if_else.at(0).first) !=
                nullptr);
    EXPECT_TRUE(dynamic_cast<sdfg::structured_control_flow::Sequence*>(&des_if_else.at(1).first) !=
                nullptr);
    auto& des_true_case =
        dynamic_cast<sdfg::structured_control_flow::Sequence&>(des_if_else.at(0).first);
    auto& des_false_case =
        dynamic_cast<sdfg::structured_control_flow::Sequence&>(des_if_else.at(1).first);
    EXPECT_EQ(des_true_case.size(), 1);
    EXPECT_EQ(des_false_case.size(), 1);
    EXPECT_TRUE(dynamic_cast<sdfg::structured_control_flow::Block*>(&des_true_case.at(0).first) !=
                nullptr);
    EXPECT_TRUE(dynamic_cast<sdfg::structured_control_flow::Block*>(&des_false_case.at(0).first) !=
                nullptr);
}

TEST(JSONSerializerTest, SerializeDeserialize_sequence) {
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    auto& scope = builder.add_sequence(root);
    auto& block1 = builder.add_block(scope);
    auto& block2 = builder.add_block(scope, {{symbolic::symbol("i"), symbolic::integer(0)}});

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the Sequence node to JSON
    nlohmann::json j;
    serializer.sequence_to_json(j, scope);

    // Deserialize the JSON back into a Sequence node
    auto des_builder = sdfg::builder::StructuredSDFGBuilder("test_sdfg", FunctionType_CPU);
    des_builder.add_container("i", sym_desc);
    serializer.json_to_sequence(j, des_builder, des_builder.subject().root());
    auto des_sdfg = des_builder.move();
    EXPECT_EQ(des_sdfg->name(), sdfg->name());
    EXPECT_EQ(des_sdfg->containers().size(), 1);
    EXPECT_EQ(des_sdfg->root().size(), 2);

    EXPECT_TRUE(dynamic_cast<sdfg::structured_control_flow::Block*>(
                    &des_sdfg->root().at(0).first) != nullptr);
    EXPECT_TRUE(dynamic_cast<sdfg::structured_control_flow::Block*>(
                    &des_sdfg->root().at(1).first) != nullptr);
    auto& transition0 = des_sdfg->root().at(0).second;
    auto& transition1 = des_sdfg->root().at(1).second;
    EXPECT_TRUE(transition0.empty());
    EXPECT_EQ(transition1.size(), 1);
    auto& assignment = transition1.assignments();
    EXPECT_EQ(assignment.size(), 1);
    EXPECT_TRUE(symbolic::eq(assignment.begin()->first, symbolic::symbol("i")));
    EXPECT_TRUE(symbolic::eq(assignment.begin()->second, symbolic::integer(0)));
}

TEST(JSONSerializerTest, SerializeDeserialize_while) {
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    auto& scope = builder.add_sequence(root);
    auto& while1 = builder.add_while(scope);
    auto& block1 = builder.add_block(while1.root());

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the Sequence node to JSON
    nlohmann::json j;
    serializer.while_node_to_json(j, while1);

    // Deserialize the JSON back into a Sequence node
    auto des_builder = sdfg::builder::StructuredSDFGBuilder("test_sdfg", FunctionType_CPU);
    des_builder.add_container("i", sym_desc);

    control_flow::Assignments assignments;

    serializer.json_to_while_node(j, des_builder, des_builder.subject().root(), assignments);
    auto des_sdfg = des_builder.move();
    EXPECT_EQ(des_sdfg->name(), sdfg->name());
    EXPECT_EQ(des_sdfg->containers().size(), 1);
    EXPECT_EQ(des_sdfg->root().size(), 1);
    EXPECT_TRUE(dynamic_cast<sdfg::structured_control_flow::While*>(
                    &des_sdfg->root().at(0).first) != nullptr);
    auto& des_while =
        dynamic_cast<sdfg::structured_control_flow::While&>(des_sdfg->root().at(0).first);

    EXPECT_EQ(des_while.root().size(), 1);
    EXPECT_TRUE(dynamic_cast<sdfg::structured_control_flow::Block*>(
                    &des_while.root().at(0).first) != nullptr);
}

TEST(JSONSerializerTest, SerializeDeserialize_while_break) {
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    auto& scope = builder.add_sequence(root);
    auto& while1 = builder.add_while(scope);
    auto& break1 = builder.add_break(while1.root());

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the Sequence node to JSON
    nlohmann::json j;
    serializer.while_node_to_json(j, while1);

    // Deserialize the JSON back into a Sequence node
    auto des_builder = sdfg::builder::StructuredSDFGBuilder("test_sdfg", FunctionType_CPU);
    des_builder.add_container("i", sym_desc);

    control_flow::Assignments assignments;

    serializer.json_to_while_node(j, des_builder, des_builder.subject().root(), assignments);
    auto des_sdfg = des_builder.move();
    EXPECT_EQ(des_sdfg->name(), sdfg->name());
    EXPECT_EQ(des_sdfg->containers().size(), 1);
    EXPECT_EQ(des_sdfg->root().size(), 1);
    EXPECT_TRUE(dynamic_cast<sdfg::structured_control_flow::While*>(
                    &des_sdfg->root().at(0).first) != nullptr);
    auto& des_while =
        dynamic_cast<sdfg::structured_control_flow::While&>(des_sdfg->root().at(0).first);
    EXPECT_TRUE(dynamic_cast<sdfg::structured_control_flow::Break*>(
                    &des_while.root().at(0).first) != nullptr);

    EXPECT_EQ(des_while.root().size(), 1);
    auto& des_break =
        dynamic_cast<sdfg::structured_control_flow::Break&>(des_while.root().at(0).first);
}

TEST(JSONSerializerTest, SerializeDeserialize_while_continue) {
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    auto& scope = builder.add_sequence(root);
    auto& while1 = builder.add_while(scope);
    auto& continue1 = builder.add_continue(while1.root());

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the Sequence node to JSON
    nlohmann::json j;
    serializer.while_node_to_json(j, while1);

    // Deserialize the JSON back into a Sequence node
    auto des_builder = sdfg::builder::StructuredSDFGBuilder("test_sdfg", FunctionType_CPU);
    des_builder.add_container("i", sym_desc);

    control_flow::Assignments assignments;

    serializer.json_to_while_node(j, des_builder, des_builder.subject().root(), assignments);
    auto des_sdfg = des_builder.move();
    EXPECT_EQ(des_sdfg->name(), sdfg->name());
    EXPECT_EQ(des_sdfg->containers().size(), 1);
    EXPECT_EQ(des_sdfg->root().size(), 1);
    EXPECT_TRUE(dynamic_cast<sdfg::structured_control_flow::While*>(
                    &des_sdfg->root().at(0).first) != nullptr);
    auto& des_while =
        dynamic_cast<sdfg::structured_control_flow::While&>(des_sdfg->root().at(0).first);
    EXPECT_TRUE(dynamic_cast<sdfg::structured_control_flow::Continue*>(
                    &des_while.root().at(0).first) != nullptr);
    EXPECT_EQ(des_while.root().size(), 1);
    auto& des_continue =
        dynamic_cast<sdfg::structured_control_flow::Continue&>(des_while.root().at(0).first);
}

TEST(JSONSerializerTest, SerializeDeserialize_Map) {
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    auto& map = builder.add_map(root, symbolic::symbol("i"), symbolic::integer(10),
                                structured_control_flow::ScheduleType_Sequential);

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the Sequence node to JSON
    nlohmann::json j;
    serializer.map_to_json(j, map);

    // Deserialize the JSON back into a Sequence node
    auto des_builder = sdfg::builder::StructuredSDFGBuilder("test_sdfg", FunctionType_CPU);

    control_flow::Assignments assignments;

    serializer.json_to_map_node(j, des_builder, des_builder.subject().root(), assignments);
    auto des_sdfg = des_builder.move();
    EXPECT_EQ(des_sdfg->name(), sdfg->name());
    EXPECT_EQ(des_sdfg->containers().size(), 0);
    EXPECT_EQ(des_sdfg->root().size(), 1);

    EXPECT_TRUE(dynamic_cast<sdfg::structured_control_flow::Map*>(&des_sdfg->root().at(0).first) !=
                nullptr);
    auto& des_map = dynamic_cast<sdfg::structured_control_flow::Map&>(des_sdfg->root().at(0).first);
    EXPECT_TRUE(symbolic::eq(des_map.indvar(), symbolic::symbol("i")));
    EXPECT_TRUE(symbolic::eq(des_map.num_iterations(), symbolic::integer(10)));

    EXPECT_EQ(des_map.root().size(), 0);
}

TEST(JSONSerializerTest, SerializeDeserialize_return) {
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    auto& ret = builder.add_return(root);

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the Sequence node to JSON
    nlohmann::json j;
    serializer.return_node_to_json(j, ret);

    // Deserialize the JSON back into a Sequence node
    auto des_builder = sdfg::builder::StructuredSDFGBuilder("test_sdfg", FunctionType_CPU);

    control_flow::Assignments assignments;

    serializer.json_to_return_node(j, des_builder, des_builder.subject().root(), assignments);
    auto des_sdfg = des_builder.move();
    EXPECT_EQ(des_sdfg->name(), sdfg->name());
    EXPECT_EQ(des_sdfg->containers().size(), 0);
    EXPECT_EQ(des_sdfg->root().size(), 1);

    EXPECT_TRUE(dynamic_cast<sdfg::structured_control_flow::Return*>(
                    &des_sdfg->root().at(0).first) != nullptr);
}

TEST(JSONSerializerTest, SerializeDeserialize) {
    // Create a sample StructuredSDFG object
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto sdfg = builder.move();

    sdfg->add_metadata("key", "value");

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the SDFG to JSON
    auto j = serializer.serialize(sdfg);

    // Deserialize the JSON back into a StructuredSDFG object
    auto sdfg_new = serializer.deserialize(j);

    // Check if the deserialized SDFG matches the original SDFG
    EXPECT_EQ(sdfg_new->name(), "test_sdfg");
    EXPECT_EQ(sdfg_new->metadata("key"), "value");
    EXPECT_EQ(sdfg_new->type(), FunctionType_CPU);
}

TEST(JSONSerializerTest, SerializeDeserialize_Arguments) {
    // Create a sample StructuredSDFG object
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);

    builder.add_container("A", base_desc, true);
    builder.add_container("C", base_desc, false);
    builder.add_container("B", base_desc, true);

    auto sdfg = builder.move();

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the SDFG to JSON
    auto j = serializer.serialize(sdfg);

    // Deserialize the JSON back into a StructuredSDFG object
    auto sdfg_new = serializer.deserialize(j);

    // Check if the deserialized SDFG matches the original SDFG
    EXPECT_EQ(sdfg_new->name(), "test_sdfg");
    EXPECT_EQ(sdfg_new->containers().size(), 3);
    EXPECT_EQ(sdfg_new->type(), FunctionType_CPU);

    EXPECT_EQ(sdfg_new->arguments().size(), 2);
    EXPECT_EQ(sdfg_new->arguments().at(0), "A");
    EXPECT_EQ(sdfg_new->arguments().at(1), "B");

    EXPECT_EQ(sdfg_new->root().size(), 0);
}

inline constexpr data_flow::LibraryNodeCode BARRIER_LOCAL{"barrier_local"};
class BarrierLocalLibraryNode : public data_flow::LibraryNode {
   public:
    BarrierLocalLibraryNode(const DebugInfo& debug_info, const graph::Vertex vertex,
                            data_flow::DataFlowGraph& parent,
                            const data_flow::LibraryNodeCode& code,
                            const std::vector<std::string>& outputs,
                            const std::vector<std::string>& inputs, const bool side_effect)
        : data_flow::LibraryNode(debug_info, vertex, parent, code, outputs, inputs, side_effect) {}

    virtual std::unique_ptr<data_flow::DataFlowNode> clone(
        const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override {
        return std::make_unique<BarrierLocalLibraryNode>(this->debug_info(), vertex, parent,
                                                         this->code(), this->outputs(),
                                                         this->inputs(), this->side_effect());
    }

    virtual void replace(const symbolic::Expression& old_expression,
                         const symbolic::Expression& new_expression) override {
        // Do nothing
    }
};

class BarrierLocalLibraryNodeSerializer : public serializer::LibraryNodeSerializer {
   public:
    nlohmann::json serialize(const sdfg::data_flow::LibraryNode& library_node) override {
        if (library_node.code() != BARRIER_LOCAL) {
            throw std::runtime_error("Invalid library node code");
        }
        nlohmann::json j;
        j["code"] = std::string(library_node.code().value());
        return j;
    }

    data_flow::LibraryNode& deserialize(const nlohmann::json& j,
                                        sdfg::builder::StructuredSDFGBuilder& builder,
                                        sdfg::structured_control_flow::Block& parent) override {
        auto code = j["code"].get<std::string_view>();
        if (code != BARRIER_LOCAL.value()) {
            throw std::runtime_error("Invalid library node code");
        }
        return builder.add_library_node<BarrierLocalLibraryNode>(parent, BARRIER_LOCAL, {}, {},
                                                                 false);
    };
};

TEST(JSONSerializerTest, SerializeDeserialize_LibraryNode) {
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    auto& block = builder.add_block(root);
    auto& lib_node =
        builder.add_library_node<BarrierLocalLibraryNode>(block, BARRIER_LOCAL, {}, {}, false);

    // Register the library node serializer
    serializer::LibraryNodeSerializerRegistry::instance().register_library_node_serializer(
        BARRIER_LOCAL.value(),
        []() { return std::make_unique<BarrierLocalLibraryNodeSerializer>(); });

    // Get the library node serializer
    auto lib_node_serializer_fn =
        serializer::LibraryNodeSerializerRegistry::instance().get_library_node_serializer(
            BARRIER_LOCAL.value());
    EXPECT_TRUE(lib_node_serializer_fn != nullptr);
    auto lib_node_serializer_ptr = lib_node_serializer_fn();
    EXPECT_TRUE(lib_node_serializer_ptr != nullptr);

    // Serialize the library node
    auto j = lib_node_serializer_ptr->serialize(lib_node);

    // Deserialize the library node
    auto& lib_node_new = lib_node_serializer_ptr->deserialize(j, builder, block);

    EXPECT_EQ(lib_node_new.code(), BARRIER_LOCAL);
    EXPECT_EQ(lib_node_new.side_effect(), lib_node.side_effect());
    EXPECT_EQ(lib_node_new.outputs(), lib_node.outputs());
    EXPECT_EQ(lib_node_new.inputs(), lib_node.inputs());

    EXPECT_TRUE(dynamic_cast<BarrierLocalLibraryNode*>(&lib_node_new));
    auto barrier_local_node = dynamic_cast<BarrierLocalLibraryNode*>(&lib_node_new);
}
