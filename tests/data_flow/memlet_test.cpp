#include <gtest/gtest.h>

#include "sdfg/builder/sdfg_builder.h"
using namespace sdfg;

TEST(MemletTest, BasicComputationalMemlet) {
    builder::SDFGBuilder builder("sdfg_memlet_basic", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Float);
    builder.add_container("input", desc);
    builder.add_container("output", desc);

    auto& input = builder.add_access(state, "input");
    auto& output = builder.add_access(state, "output");
    auto& tasklet = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});

    builder.add_computational_memlet(state, input, tasklet, "_in", {});
    builder.add_computational_memlet(state, tasklet, "_out", output, {});

    // Check memlet exists
    auto& graph = state.dataflow();
    EXPECT_EQ(graph.in_degree(tasklet), 1);
    EXPECT_EQ(graph.out_degree(tasklet), 1);
}

TEST(MemletTest, MemletType) {
    builder::SDFGBuilder builder("sdfg_memlet_type", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("x", desc);
    builder.add_container("y", desc);

    auto& x_node = builder.add_access(state, "x");
    auto& y_node = builder.add_access(state, "y");
    auto& tasklet = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});

    builder.add_computational_memlet(state, x_node, tasklet, "_in", {});
    builder.add_computational_memlet(state, tasklet, "_out", y_node, {});

    auto& graph = state.dataflow();
    for (auto& memlet : graph.in_edges(tasklet)) {
        EXPECT_EQ(memlet.type(), data_flow::MemletType::Computational);
    }
}

TEST(MemletTest, MemletSourceAndDestination) {
    builder::SDFGBuilder builder("sdfg_memlet_src_dst", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Double);
    builder.add_container("a", desc);
    builder.add_container("b", desc);

    auto& a_node = builder.add_access(state, "a");
    auto& b_node = builder.add_access(state, "b");
    auto& tasklet = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});

    builder.add_computational_memlet(state, a_node, tasklet, "_in", {});
    builder.add_computational_memlet(state, tasklet, "_out", b_node, {});

    auto& graph = state.dataflow();
    for (auto& memlet : graph.in_edges(tasklet)) {
        EXPECT_EQ(&memlet.src(), &a_node);
        EXPECT_EQ(&memlet.dst(), &tasklet);
    }

    for (auto& memlet : graph.out_edges(tasklet)) {
        EXPECT_EQ(&memlet.src(), &tasklet);
        EXPECT_EQ(&memlet.dst(), &b_node);
    }
}

TEST(MemletTest, MemletConnectors) {
    builder::SDFGBuilder builder("sdfg_memlet_conn", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Float);
    builder.add_container("input", desc);
    builder.add_container("output", desc);

    auto& input = builder.add_access(state, "input");
    auto& output = builder.add_access(state, "output");
    auto& tasklet = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});

    builder.add_computational_memlet(state, input, tasklet, "_in", {});
    builder.add_computational_memlet(state, tasklet, "_out", output, {});

    auto& graph = state.dataflow();
    for (auto& memlet : graph.in_edges(tasklet)) {
        EXPECT_EQ(memlet.src_conn(), "void");
        EXPECT_EQ(memlet.dst_conn(), "_in");
    }

    for (auto& memlet : graph.out_edges(tasklet)) {
        EXPECT_EQ(memlet.src_conn(), "_out");
        EXPECT_EQ(memlet.dst_conn(), "void");
    }
}

TEST(MemletTest, MemletSubset) {
    builder::SDFGBuilder builder("sdfg_memlet_subset", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("data", desc);
    builder.add_container("result", desc);

    auto& data_node = builder.add_access(state, "data");
    auto& result_node = builder.add_access(state, "result");
    auto& tasklet = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});

    builder.add_computational_memlet(state, data_node, tasklet, "_in", {});
    builder.add_computational_memlet(state, tasklet, "_out", result_node, {});

    auto& graph = state.dataflow();
    for (auto& memlet : graph.in_edges(tasklet)) {
        // Empty subset for scalar
        EXPECT_EQ(memlet.subset().size(), 0);
    }
}

TEST(MemletTest, MultipleMemlets) {
    builder::SDFGBuilder builder("sdfg_multi_memlets", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Double);
    builder.add_container("a", desc);
    builder.add_container("b", desc);
    builder.add_container("c", desc);

    auto& a_node = builder.add_access(state, "a");
    auto& b_node = builder.add_access(state, "b");
    auto& c_node = builder.add_access(state, "c");
    auto& tasklet = builder.add_tasklet(state, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});

    builder.add_computational_memlet(state, a_node, tasklet, "_in1", {});
    builder.add_computational_memlet(state, b_node, tasklet, "_in2", {});
    builder.add_computational_memlet(state, tasklet, "_out", c_node, {});

    auto& graph = state.dataflow();
    EXPECT_EQ(graph.in_degree(tasklet), 2);
    EXPECT_EQ(graph.out_degree(tasklet), 1);
}

TEST(MemletTest, MemletParentGraph) {
    builder::SDFGBuilder builder("sdfg_memlet_parent", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("x", desc);
    builder.add_container("y", desc);

    auto& x_node = builder.add_access(state, "x");
    auto& y_node = builder.add_access(state, "y");
    auto& tasklet = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});

    builder.add_computational_memlet(state, x_node, tasklet, "_in", {});
    builder.add_computational_memlet(state, tasklet, "_out", y_node, {});

    auto& graph = state.dataflow();
    for (auto& memlet : graph.in_edges(tasklet)) {
        EXPECT_EQ(&memlet.get_parent(), &graph);
    }
}

TEST(MemletTest, MemletEdge) {
    builder::SDFGBuilder builder("sdfg_memlet_edge", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Float);
    builder.add_container("input", desc);
    builder.add_container("output", desc);

    auto& input = builder.add_access(state, "input");
    auto& output = builder.add_access(state, "output");
    auto& tasklet = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});

    builder.add_computational_memlet(state, input, tasklet, "_in", {});
    builder.add_computational_memlet(state, tasklet, "_out", output, {});

    auto& graph = state.dataflow();
    for (auto& memlet : graph.in_edges(tasklet)) {
        auto edge = memlet.edge();
        EXPECT_NE(edge, graph::Edge());
    }
}

TEST(MemletTest, MemletBaseType) {
    builder::SDFGBuilder builder("sdfg_memlet_base_type", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Int64);
    builder.add_container("data", desc);
    builder.add_container("result", desc);

    auto& data_node = builder.add_access(state, "data");
    auto& result_node = builder.add_access(state, "result");
    auto& tasklet = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});

    builder.add_computational_memlet(state, data_node, tasklet, "_in", {});
    builder.add_computational_memlet(state, tasklet, "_out", result_node, {});

    auto& graph = state.dataflow();
    for (auto& memlet : graph.in_edges(tasklet)) {
        EXPECT_EQ(memlet.base_type().type_id(), types::TypeID::Scalar);
    }
}

TEST(MemletTest, ChainedMemlets) {
    builder::SDFGBuilder builder("sdfg_chained_memlets", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Double);
    builder.add_container("a", desc);
    builder.add_container("b", desc);
    builder.add_container("c", desc);

    auto& a_node = builder.add_access(state, "a");
    auto& b_node = builder.add_access(state, "b");
    auto& c_node = builder.add_access(state, "c");
    auto& tasklet1 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    auto& tasklet2 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});

    builder.add_computational_memlet(state, a_node, tasklet1, "_in", {});
    builder.add_computational_memlet(state, tasklet1, "_out", b_node, {});
    builder.add_computational_memlet(state, b_node, tasklet2, "_in", {});
    builder.add_computational_memlet(state, tasklet2, "_out", c_node, {});

    auto& graph = state.dataflow();
    EXPECT_EQ(graph.out_degree(a_node), 1);
    EXPECT_EQ(graph.in_degree(b_node), 1);
    EXPECT_EQ(graph.out_degree(b_node), 1);
    EXPECT_EQ(graph.in_degree(c_node), 1);
}

TEST(MemletTest, FanOutMemlets) {
    builder::SDFGBuilder builder("sdfg_fanout_memlets", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Float);
    builder.add_container("input", desc);
    builder.add_container("out1", desc);
    builder.add_container("out2", desc);

    auto& input = builder.add_access(state, "input");
    auto& out1 = builder.add_access(state, "out1");
    auto& out2 = builder.add_access(state, "out2");
    auto& tasklet1 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    auto& tasklet2 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});

    builder.add_computational_memlet(state, input, tasklet1, "_in", {});
    builder.add_computational_memlet(state, input, tasklet2, "_in", {});
    builder.add_computational_memlet(state, tasklet1, "_out", out1, {});
    builder.add_computational_memlet(state, tasklet2, "_out", out2, {});

    auto& graph = state.dataflow();
    EXPECT_EQ(graph.out_degree(input), 2);
}

TEST(MemletTest, FanInMemlets) {
    builder::SDFGBuilder builder("sdfg_fanin_memlets", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("in1", desc);
    builder.add_container("in2", desc);
    builder.add_container("output", desc);

    auto& in1 = builder.add_access(state, "in1");
    auto& in2 = builder.add_access(state, "in2");
    auto& output = builder.add_access(state, "output");
    auto& tasklet1 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    auto& tasklet2 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});

    builder.add_computational_memlet(state, in1, tasklet1, "_in", {});
    builder.add_computational_memlet(state, in2, tasklet2, "_in", {});
    builder.add_computational_memlet(state, tasklet1, "_out", output, {});
    builder.add_computational_memlet(state, tasklet2, "_out", output, {});

    auto& graph = state.dataflow();
    EXPECT_EQ(graph.in_degree(output), 2);
}

TEST(MemletTest, ConstantToMemlet) {
    builder::SDFGBuilder builder("sdfg_const_memlet", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Double);
    builder.add_container("output", desc);

    auto& constant = builder.add_constant(state, "2.718", desc);
    auto& output = builder.add_access(state, "output");
    auto& tasklet = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});

    builder.add_computational_memlet(state, constant, tasklet, "_in", {});
    builder.add_computational_memlet(state, tasklet, "_out", output, {});

    auto& graph = state.dataflow();
    EXPECT_EQ(graph.out_degree(constant), 1);
    EXPECT_EQ(graph.in_degree(tasklet), 1);
}

TEST(MemletTest, ReplaceSymbolicExpression) {
    builder::SDFGBuilder builder("sdfg_memlet_replace", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Float);
    builder.add_container("x", desc);
    builder.add_container("y", desc);

    auto& x_node = builder.add_access(state, "x");
    auto& y_node = builder.add_access(state, "y");
    auto& tasklet = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});

    builder.add_computational_memlet(state, x_node, tasklet, "_in", {});
    builder.add_computational_memlet(state, tasklet, "_out", y_node, {});

    // This test verifies replace method exists and can be called
    auto old_expr = symbolic::symbol("old");
    auto new_expr = symbolic::symbol("new");

    auto& graph = state.dataflow();
    for (auto& memlet : graph.in_edges(tasklet)) {
        EXPECT_NO_THROW(memlet.replace(old_expr, new_expr));
    }
}

TEST(MemletTest, ElementIdPresent) {
    builder::SDFGBuilder builder("sdfg_memlet_id", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::UInt8);
    builder.add_container("a", desc);
    builder.add_container("b", desc);

    auto& a_node = builder.add_access(state, "a");
    auto& b_node = builder.add_access(state, "b");
    auto& tasklet = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});

    builder.add_computational_memlet(state, a_node, tasklet, "_in", {});
    builder.add_computational_memlet(state, tasklet, "_out", b_node, {});

    auto& graph = state.dataflow();
    for (auto& memlet : graph.in_edges(tasklet)) {
        // Memlet is an Element, so it has an element_id
        EXPECT_GE(memlet.element_id(), 0);
    }
}

TEST(MemletTest, DistinctMemletIds) {
    builder::SDFGBuilder builder("sdfg_distinct_memlet_ids", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("a", desc);
    builder.add_container("b", desc);
    builder.add_container("c", desc);

    auto& a_node = builder.add_access(state, "a");
    auto& b_node = builder.add_access(state, "b");
    auto& c_node = builder.add_access(state, "c");
    auto& tasklet = builder.add_tasklet(state, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});

    builder.add_computational_memlet(state, a_node, tasklet, "_in1", {});
    builder.add_computational_memlet(state, b_node, tasklet, "_in2", {});
    builder.add_computational_memlet(state, tasklet, "_out", c_node, {});

    auto& graph = state.dataflow();
    std::set<size_t> ids;
    for (auto& memlet : graph.in_edges(tasklet)) {
        ids.insert(memlet.element_id());
    }
    for (auto& memlet : graph.out_edges(tasklet)) {
        ids.insert(memlet.element_id());
    }

    // All memlets should have distinct IDs
    EXPECT_EQ(ids.size(), 3);
}
