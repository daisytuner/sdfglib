#include <gtest/gtest.h>

#include "sdfg/builder/sdfg_builder.h"
using namespace sdfg;

TEST(DataFlowNodeTest, GetVertex) {
    builder::SDFGBuilder builder("sdfg_vertex", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Float);
    builder.add_container("data", desc);

    auto& access_node = builder.add_access(state, "data");
    auto vertex = access_node.vertex();

    // Vertex should be valid
    EXPECT_NE(vertex, graph::Vertex());
}

TEST(DataFlowNodeTest, GetParentGraph) {
    builder::SDFGBuilder builder("sdfg_parent", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Double);
    builder.add_container("var", desc);

    auto& access_node = builder.add_access(state, "var");
    auto& parent = access_node.get_parent();

    EXPECT_EQ(&parent, &state.dataflow());
}

TEST(DataFlowNodeTest, GetConstParentGraph) {
    builder::SDFGBuilder builder("sdfg_const_parent", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("data", desc);

    auto& access_node = builder.add_access(state, "data");
    const auto& const_node = access_node;
    const auto& parent = const_node.get_parent();

    EXPECT_EQ(&parent, &state.dataflow());
}

TEST(DataFlowNodeTest, MultipleNodesDistinctVertices) {
    builder::SDFGBuilder builder("sdfg_distinct_vertices", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("a", desc);
    builder.add_container("b", desc);

    auto& node_a = builder.add_access(state, "a");
    auto& node_b = builder.add_access(state, "b");

    EXPECT_NE(node_a.vertex(), node_b.vertex());
}

TEST(DataFlowNodeTest, TaskletVertexDistinct) {
    builder::SDFGBuilder builder("sdfg_tasklet_vertex", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Float);
    builder.add_container("input", desc);
    builder.add_container("output", desc);

    auto& input_node = builder.add_access(state, "input");
    auto& output_node = builder.add_access(state, "output");
    auto& tasklet = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});

    builder.add_computational_memlet(state, input_node, tasklet, "_in", {});
    builder.add_computational_memlet(state, tasklet, "_out", output_node, {});

    EXPECT_NE(input_node.vertex(), tasklet.vertex());
    EXPECT_NE(output_node.vertex(), tasklet.vertex());
    EXPECT_NE(input_node.vertex(), output_node.vertex());
}

TEST(DataFlowNodeTest, ConstantNodeVertex) {
    builder::SDFGBuilder builder("sdfg_const_vertex", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Int32);
    auto& constant = builder.add_constant(state, "42", desc);

    auto vertex = constant.vertex();
    EXPECT_NE(vertex, graph::Vertex());
}

TEST(DataFlowNodeTest, AccessNodeInheritance) {
    builder::SDFGBuilder builder("sdfg_inheritance", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Double);
    builder.add_container("data", desc);

    auto& access_node = builder.add_access(state, "data");

    // AccessNode is a DataFlowNode
    data_flow::DataFlowNode& base_ref = access_node;
    EXPECT_EQ(&base_ref, &access_node);
}

TEST(DataFlowNodeTest, TaskletInheritance) {
    builder::SDFGBuilder builder("sdfg_tasklet_inherit", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Float);
    builder.add_container("in", desc);
    builder.add_container("out", desc);

    auto& input = builder.add_access(state, "in");
    auto& output = builder.add_access(state, "out");
    auto& tasklet = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});

    builder.add_computational_memlet(state, input, tasklet, "_in", {});
    builder.add_computational_memlet(state, tasklet, "_out", output, {});

    // Tasklet is a DataFlowNode
    data_flow::DataFlowNode& base_ref = tasklet;
    EXPECT_EQ(&base_ref, &tasklet);
}

TEST(DataFlowNodeTest, ConstantNodeInheritance) {
    builder::SDFGBuilder builder("sdfg_const_inherit", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Int64);
    auto& constant = builder.add_constant(state, "100", desc);

    // ConstantNode is a DataFlowNode
    data_flow::DataFlowNode& base_ref = constant;
    EXPECT_EQ(&base_ref, &constant);
}

TEST(DataFlowNodeTest, ElementIdPresent) {
    builder::SDFGBuilder builder("sdfg_element_id", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Bool);
    builder.add_container("flag", desc);

    auto& access_node = builder.add_access(state, "flag");

    // DataFlowNode is an Element, so it has an element_id
    EXPECT_GE(access_node.element_id(), 0);
}

TEST(DataFlowNodeTest, MultipleNodesDistinctIds) {
    builder::SDFGBuilder builder("sdfg_distinct_ids", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::UInt8);
    builder.add_container("x", desc);
    builder.add_container("y", desc);
    builder.add_container("z", desc);

    auto& node_x = builder.add_access(state, "x");
    auto& node_y = builder.add_access(state, "y");
    auto& node_z = builder.add_access(state, "z");

    EXPECT_NE(node_x.element_id(), node_y.element_id());
    EXPECT_NE(node_y.element_id(), node_z.element_id());
    EXPECT_NE(node_x.element_id(), node_z.element_id());
}

TEST(DataFlowNodeTest, SameParentMultipleNodes) {
    builder::SDFGBuilder builder("sdfg_same_parent", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Float);
    builder.add_container("a", desc);
    builder.add_container("b", desc);

    auto& node_a = builder.add_access(state, "a");
    auto& node_b = builder.add_access(state, "b");

    EXPECT_EQ(&node_a.get_parent(), &node_b.get_parent());
    EXPECT_EQ(&node_a.get_parent(), &state.dataflow());
}

TEST(DataFlowNodeTest, VertexInGraph) {
    builder::SDFGBuilder builder("sdfg_vertex_in_graph", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("data", desc);

    auto& access_node = builder.add_access(state, "data");
    auto& graph = state.dataflow();

    // Verify the vertex exists in the graph by checking we can iterate nodes
    bool found = false;
    for (auto& node : graph.nodes()) {
        if (&node == &access_node) {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found);
}

TEST(DataFlowNodeTest, DebugInfo) {
    builder::SDFGBuilder builder("sdfg_debug_info", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Double);
    builder.add_container("value", desc);

    DebugInfo debug_info;
    debug_info.file = "test.cpp";
    debug_info.line = 42;
    debug_info.column = 10;

    auto& access_node = builder.add_access(state, "value", debug_info);

    EXPECT_EQ(access_node.debug_info().file, "test.cpp");
    EXPECT_EQ(access_node.debug_info().line, 42);
    EXPECT_EQ(access_node.debug_info().column, 10);
}
