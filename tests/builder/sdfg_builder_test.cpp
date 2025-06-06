#include "sdfg/builder/sdfg_builder.h"

#include <gtest/gtest.h>

#include "sdfg/data_flow/library_node.h"
using namespace sdfg;

TEST(SDFGBuilderTest, Empty) {
    builder::SDFGBuilder builder("sdfg_1", FunctionType::CPU);

    auto sdfg = builder.move();

    EXPECT_EQ(sdfg->name(), "sdfg_1");
    EXPECT_EQ(sdfg->states().size(), 0);
    EXPECT_EQ(sdfg->edges().size(), 0);
}

TEST(SDFGBuilderTest, AddState) {
    builder::SDFGBuilder builder("sdfg_1", FunctionType::CPU);

    auto& state = builder.add_state();

    auto sdfg = builder.move();

    auto states = sdfg->states();
    EXPECT_EQ(states.size(), 1);
    EXPECT_EQ(&*states.begin(), &state);

    EXPECT_EQ(sdfg->in_degree(state), 0);
    EXPECT_EQ(sdfg->out_degree(state), 0);
}

TEST(SDFGBuilderTest, AddStartState) {
    builder::SDFGBuilder builder("sdfg_1", FunctionType::CPU);

    auto& state = builder.add_state(true);

    auto sdfg = builder.move();

    auto states = sdfg->states();
    EXPECT_EQ(states.size(), 1);
    EXPECT_EQ(&*states.begin(), &state);

    EXPECT_EQ(sdfg->in_degree(state), 0);
    EXPECT_EQ(sdfg->out_degree(state), 0);

    EXPECT_EQ(&sdfg->start_state(), &state);
}

TEST(SDFGBuilderTest, AddStateBefore) {
    builder::SDFGBuilder builder("sdfg_1", FunctionType::CPU);

    auto& state_1 = builder.add_state();
    auto& state_2 = builder.add_state();
    auto& edge_1 = builder.add_edge(state_1, state_2);

    auto& state_3 = builder.add_state_before(state_2);

    auto sdfg = builder.move();

    auto states = sdfg->states();
    EXPECT_EQ(states.size(), 3);

    EXPECT_EQ(sdfg->in_degree(state_1), 0);
    EXPECT_EQ(sdfg->out_degree(state_1), 1);
    EXPECT_EQ(sdfg->in_degree(state_2), 1);
    EXPECT_EQ(sdfg->out_degree(state_2), 0);
    EXPECT_EQ(sdfg->in_degree(state_3), 1);
    EXPECT_EQ(sdfg->out_degree(state_3), 1);
}

TEST(SDFGBuilderTest, AddStateAfter) {
    builder::SDFGBuilder builder("sdfg_1", FunctionType::CPU);

    auto& state_1 = builder.add_state();
    auto& state_2 = builder.add_state();
    auto& edge_1 = builder.add_edge(state_1, state_2);

    auto& state_3 = builder.add_state_after(state_1);

    auto sdfg = builder.move();

    auto states = sdfg->states();
    EXPECT_EQ(states.size(), 3);

    EXPECT_EQ(sdfg->in_degree(state_1), 0);
    EXPECT_EQ(sdfg->out_degree(state_1), 1);
    EXPECT_EQ(sdfg->in_degree(state_2), 1);
    EXPECT_EQ(sdfg->out_degree(state_2), 0);
    EXPECT_EQ(sdfg->in_degree(state_3), 1);
    EXPECT_EQ(sdfg->out_degree(state_3), 1);
}

TEST(SDFGBuilderTest, AddEdge) {
    builder::SDFGBuilder builder("sdfg_1", FunctionType::CPU);

    auto& state_1 = builder.add_state();
    auto& state_2 = builder.add_state();
    auto& edge = builder.add_edge(state_1, state_2);
    EXPECT_EQ(&edge.src(), &state_1);
    EXPECT_EQ(&edge.dst(), &state_2);
    EXPECT_TRUE(symbolic::is_true(edge.condition()));

    auto sdfg = builder.move();

    auto edges = sdfg->edges();
    EXPECT_EQ(edges.size(), 1);
    EXPECT_EQ(&*edges.begin(), &edge);

    EXPECT_EQ(sdfg->in_degree(state_1), 0);
    EXPECT_EQ(sdfg->out_degree(state_1), 1);
    EXPECT_EQ(&*sdfg->out_edges(state_1).begin(), &edge);

    EXPECT_EQ(sdfg->in_degree(state_2), 1);
    EXPECT_EQ(sdfg->out_degree(state_2), 0);
    EXPECT_EQ(&*sdfg->in_edges(state_2).begin(), &edge);
}

TEST(SDFGBuilderTest, addEdgeWithCondition) {
    builder::SDFGBuilder builder("sdfg_1", FunctionType::CPU);

    builder.add_container("i", types::Scalar(types::PrimitiveType::UInt64));
    symbolic::Symbol iter_sym = sdfg::symbolic::symbol("i");

    auto cond = symbolic::Eq(iter_sym, SymEngine::integer(0));

    auto& state_guard = builder.add_state();
    auto& state_if = builder.add_state();
    auto& state_else = builder.add_state();
    auto& edge_if = builder.add_edge(state_guard, state_if, cond);
    auto& edge_else = builder.add_edge(state_guard, state_else, symbolic::Not(cond));

    EXPECT_EQ(&edge_if.src(), &state_guard);
    EXPECT_EQ(&edge_if.dst(), &state_if);
    EXPECT_EQ(edge_if.condition()->__str__(), "0 == i");
    EXPECT_EQ(&edge_else.src(), &state_guard);
    EXPECT_EQ(&edge_else.dst(), &state_else);
    EXPECT_EQ(edge_else.condition()->__str__(), "0 != i");

    auto sdfg = builder.move();
}

TEST(SDFGBuilderTest, addEdgeWithAssignments) {
    builder::SDFGBuilder builder("sdfg_1", FunctionType::CPU);

    builder.add_container("i", types::Scalar(types::PrimitiveType::UInt64));
    symbolic::Symbol iter_sym = sdfg::symbolic::symbol("i");

    auto& state_1 = builder.add_state();
    auto& state_2 = builder.add_state();
    auto& edge = builder.add_edge(state_1, state_2,
                                  symbolic::Assignments{{iter_sym, SymEngine::integer(0)}});

    EXPECT_EQ(&edge.src(), &state_1);
    EXPECT_EQ(&edge.dst(), &state_2);
    EXPECT_TRUE(edge.is_unconditional());
    EXPECT_EQ(edge.assignments().size(), 1);

    auto assignment = edge.assignments().begin();
    EXPECT_EQ(assignment->first->get_name(), "i");
    EXPECT_EQ(assignment->second->__str__(), "0");

    auto sdfg = builder.move();
}

TEST(SDFGBuilderTest, AddAccessNode) {
    builder::SDFGBuilder builder("sdfg_1", FunctionType::CPU);

    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Double);
    builder.add_container("scalar_1", desc);

    auto& access_node = builder.add_access(state, "scalar_1");

    auto sdfg = builder.move();

    EXPECT_EQ(state.dataflow().nodes().size(), 1);
    EXPECT_EQ(state.dataflow().edges().size(), 0);
    EXPECT_EQ(access_node.data(), "scalar_1");
}

TEST(SDFGBuilderTest, AddTasklet) {
    builder::SDFGBuilder builder("sdfg_1", FunctionType::CPU);

    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Double);
    builder.add_container("scalar_1", desc);

    auto& access_node_in = builder.add_access(state, "scalar_1");
    auto& access_node_out = builder.add_access(state, "scalar_1");
    auto& tasklet =
        builder.add_tasklet(state, data_flow::TaskletCode::assign, {"_out", desc}, {{"_in", desc}});

    builder.add_memlet(state, access_node_in, "void", tasklet, "_in", {});
    builder.add_memlet(state, tasklet, "_out", access_node_out, "void", {});

    auto sdfg = builder.move();

    EXPECT_EQ(state.dataflow().nodes().size(), 3);
    EXPECT_EQ(state.dataflow().edges().size(), 2);
    EXPECT_EQ(tasklet.code(), data_flow::TaskletCode::assign);
}

TEST(SDFGBuilderTest, AddLibnode) {
    builder::SDFGBuilder builder("sdfg_1", FunctionType::CPU);

    auto& state = builder.add_state(true);

    builder.add_library_node(state, data_flow::LibraryNodeType::barrier_local, {}, {}, false);

    auto sdfg = builder.move();
    auto states = sdfg->states();
    EXPECT_EQ(states.size(), 1);
    EXPECT_EQ(&*states.begin(), &state);

    EXPECT_EQ(sdfg->in_degree(state), 0);
    EXPECT_EQ(sdfg->out_degree(state), 0);

    EXPECT_EQ(&sdfg->start_state(), &state);
}
