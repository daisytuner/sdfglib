#include "sdfg/serializer/json_serializer.h"

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/kernel.h"
#include "sdfg/structured_control_flow/return.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/while.h"

using namespace sdfg;

TEST(JSONSerializerTest, DatatypeToJSON_Scalar) {
    // Create a sample data type
    types::Scalar scalar_type(types::PrimitiveType::Int32);
    nlohmann::json j;

    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg");
    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer(filename, sdfg);

    // Serialize the data type to JSON
    serializer.type_to_json(j, scalar_type);

    // Check if the JSON contains the expected keys
    EXPECT_TRUE(j.contains("type"));
    EXPECT_EQ(j["type"], "scalar");
    EXPECT_TRUE(j.contains("primitive_type"));
    EXPECT_EQ(j["primitive_type"], scalar_type.primitive_type());
    EXPECT_TRUE(j.contains("address_space"));
    EXPECT_EQ(j["address_space"], scalar_type.address_space());
    EXPECT_TRUE(j.contains("initializer"));
    EXPECT_EQ(j["initializer"], scalar_type.initializer());
    EXPECT_TRUE(j.contains("device_location"));
    EXPECT_EQ(j["device_location"], scalar_type.device_location());
}

TEST(JSONSerializerTest, DatatypeToJSON_Pointer) {
    // Create a sample data type
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer pointer_type(base_desc);
    nlohmann::json j;

    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg");
    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer(filename, sdfg);

    // Serialize the data type to JSON
    serializer.type_to_json(j, pointer_type);

    // Check if the JSON contains the expected keys
    EXPECT_TRUE(j.contains("type"));
    EXPECT_EQ(j["type"], "pointer");
    EXPECT_TRUE(j.contains("pointee_type"));
    EXPECT_EQ(j["pointee_type"]["type"], "scalar");
    EXPECT_EQ(j["pointee_type"]["primitive_type"], base_desc.primitive_type());
    EXPECT_TRUE(j.contains("address_space"));
    EXPECT_EQ(j["address_space"], pointer_type.address_space());
    EXPECT_TRUE(j.contains("initializer"));
    EXPECT_EQ(j["initializer"], pointer_type.initializer());
    EXPECT_TRUE(j.contains("device_location"));
    EXPECT_EQ(j["device_location"], pointer_type.device_location());
}

TEST(JSONSerializerTest, DatatypeToJSON_Structure) {
    // Create a sample data type
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Structure structure_type("MyStruct");
    nlohmann::json j;

    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg");
    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer(filename, sdfg);

    // Serialize the data type to JSON
    serializer.type_to_json(j, structure_type);

    // Check if the JSON contains the expected keys
    EXPECT_TRUE(j.contains("type"));
    EXPECT_EQ(j["type"], "structure");
    EXPECT_TRUE(j.contains("name"));
    EXPECT_EQ(j["name"], "MyStruct");
    EXPECT_TRUE(j.contains("address_space"));
    EXPECT_EQ(j["address_space"], structure_type.address_space());
    EXPECT_TRUE(j.contains("initializer"));
    EXPECT_EQ(j["initializer"], structure_type.initializer());
    EXPECT_TRUE(j.contains("device_location"));
    EXPECT_EQ(j["device_location"], structure_type.device_location());
}

TEST(JSONSerializerTest, DatatypeToJSON_Array) {
    // Create a sample data type
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_type(base_desc, {symbolic::symbol("N")});
    nlohmann::json j;

    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg");
    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer(filename, sdfg);

    // Serialize the data type to JSON
    serializer.type_to_json(j, array_type);

    // Check if the JSON contains the expected keys
    EXPECT_TRUE(j.contains("type"));
    EXPECT_EQ(j["type"], "array");
    EXPECT_TRUE(j.contains("element_type"));
    EXPECT_EQ(j["element_type"]["type"], "scalar");
    EXPECT_EQ(j["element_type"]["primitive_type"], base_desc.primitive_type());

    EXPECT_TRUE(j.contains("num_elements"));
    EXPECT_EQ(j["num_elements"], "N");

    EXPECT_TRUE(j.contains("address_space"));
    EXPECT_EQ(j["address_space"], array_type.address_space());
    EXPECT_TRUE(j.contains("initializer"));
    EXPECT_EQ(j["initializer"], array_type.initializer());
    EXPECT_TRUE(j.contains("device_location"));
    EXPECT_EQ(j["device_location"], array_type.device_location());
}

TEST(JSONSerializerTest, DataflowToJSON) {
    // Create a sample Block object
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg");
    auto& block = builder.add_block(builder.subject().root());

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
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
    auto& memlet_out = builder.add_memlet(block, tasklet, "_out", access_out, "void", {});

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer(filename, sdfg);

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
        return node["type"] == "access_node" && node["container"] == "A";
    });
    auto access_node_C = std::find_if(j["nodes"].begin(), j["nodes"].end(), [](const auto& node) {
        return node["type"] == "access_node" && node["container"] == "C";
    });
    auto access_node_D = std::find_if(j["nodes"].begin(), j["nodes"].end(), [](const auto& node) {
        return node["type"] == "access_node" && node["container"] == "D";
    });
    auto tasklet_node = std::find_if(j["nodes"].begin(), j["nodes"].end(), [](const auto& node) {
        return node["type"] == "tasklet" && node["code"] == data_flow::TaskletCode::add;
    });

    EXPECT_NE(access_node_A, j["nodes"].end());
    EXPECT_NE(access_node_C, j["nodes"].end());
    EXPECT_NE(access_node_D, j["nodes"].end());
    EXPECT_NE(tasklet_node, j["nodes"].end());

    EXPECT_EQ(tasklet_node->at("inputs").size(), 2);
    EXPECT_EQ(tasklet_node->at("outputs").size(), 1);

    auto edge_to_tasklet =
        std::find_if(j["edges"].begin(), j["edges"].end(), [&](const auto& edge) {
            return edge["source"] == access_node_A->at("element_id") &&
                   edge["target"] == tasklet_node->at("element_id") &&
                   edge["source_connector"] == "void" && edge["target_connector"] == "_in1";
        });
    auto edge_to_tasklet2 =
        std::find_if(j["edges"].begin(), j["edges"].end(), [&](const auto& edge) {
            return edge["source"] == access_node_C->at("element_id") &&
                   edge["target"] == tasklet_node->at("element_id") &&
                   edge["source_connector"] == "void" && edge["target_connector"] == "_in2";
        });
    auto edge_from_tasklet =
        std::find_if(j["edges"].begin(), j["edges"].end(), [&](const auto& edge) {
            return edge["source"] == tasklet_node->at("element_id") &&
                   edge["target"] == access_node_D->at("element_id") &&
                   edge["source_connector"] == "_out" && edge["target_connector"] == "void";
        });

    std::cout << j.dump(2) << std::endl;

    EXPECT_NE(edge_to_tasklet, j["edges"].end());
    EXPECT_NE(edge_to_tasklet2, j["edges"].end());
    EXPECT_NE(edge_from_tasklet, j["edges"].end());
}

TEST(JSONSerializerTest, BlockToJSON) {
    // Create a sample Block object
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg");
    auto& block = builder.add_block(builder.subject().root());

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer(filename, sdfg);

    // Serialize the Block to JSON
    nlohmann::json j;
    serializer.block_to_json(j, block);

    // Check if the JSON contains the expected keys
    EXPECT_TRUE(j.contains("type"));
    EXPECT_EQ(j["type"], "block");
}

TEST(JSONSerializerTest, ForNodeToJSON) {
    // Create a sample For node
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg");
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
    sdfg::serializer::JSONSerializer serializer(filename, sdfg);

    // Serialize the For node to JSON
    nlohmann::json j;
    serializer.for_node_to_json(j, scope);

    // Check if the JSON contains the expected keys
    EXPECT_TRUE(j.contains("type"));
    EXPECT_EQ(j["type"], "for");
}

TEST(JSONSerializerTest, SerializeDeserialize) {
    // Create a sample StructuredSDFG object
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg");
    auto sdfg = builder.move();

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    sdfg::serializer::JSONSerializer serializer(filename, sdfg);

    // Serialize the SDFG to JSON
    serializer.serialize();

    // Deserialize the JSON back into a StructuredSDFG object
    auto sdfg_new = serializer.deserialize();

    // Check if the deserialized SDFG matches the original SDFG
    EXPECT_EQ(sdfg_new->name(), "test_sdfg");
}