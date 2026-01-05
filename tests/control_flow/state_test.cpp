#include "sdfg/control_flow/state.h"

#include <gtest/gtest.h>

#include "sdfg/builder/sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/exceptions.h"
#include "sdfg/types/scalar.h"

using namespace sdfg;

/**
 * Test fixture for State tests
 */
class StateTest : public ::testing::Test {
};

/**
 * Test basic state creation and properties
 */
TEST_F(StateTest, BasicStateCreation) {
    builder::SDFGBuilder builder("test_sdfg", FunctionType_CPU);

    auto& state = builder.add_state();

    // Check that state has valid element ID
    EXPECT_GT(state.element_id(), 0);

    // Check that dataflow graph is initially empty
    EXPECT_EQ(state.dataflow().nodes().size(), 0);
    EXPECT_EQ(state.dataflow().edges().size(), 0);
}

/**
 * Test that state can be used as start state
 */
TEST_F(StateTest, StartState) {
    builder::SDFGBuilder builder("test_sdfg", FunctionType_CPU);

    auto& state = builder.add_state(true);
    auto& sdfg = builder.subject();

    EXPECT_EQ(&sdfg.start_state(), &state);
}

/**
 * Test that state can contain data-flow nodes
 */
TEST_F(StateTest, StateWithDataFlowNodes) {
    builder::SDFGBuilder builder("test_sdfg", FunctionType_CPU);

    builder.add_container("x", types::Scalar(types::PrimitiveType::Double));

    auto& state = builder.add_state();
    auto& access_node = builder.add_access(state, "x");

    // Verify the dataflow graph contains the node
    EXPECT_EQ(state.dataflow().nodes().size(), 1);
    EXPECT_EQ(&*state.dataflow().nodes().begin(), &access_node);
}

/**
 * Test that state can contain tasklets and edges
 */
TEST_F(StateTest, StateWithTasklet) {
    builder::SDFGBuilder builder("test_sdfg", FunctionType_CPU);

    builder.add_container("a", types::Scalar(types::PrimitiveType::Double));
    builder.add_container("b", types::Scalar(types::PrimitiveType::Double));

    auto& state = builder.add_state();
    auto& access_in = builder.add_access(state, "a");
    auto& tasklet = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    auto& access_out = builder.add_access(state, "b");

    builder.add_computational_memlet(state, access_in, tasklet, "_in", {});
    builder.add_computational_memlet(state, tasklet, "_out", access_out, {});

    // Verify the dataflow graph structure
    EXPECT_EQ(state.dataflow().nodes().size(), 3);
    EXPECT_EQ(state.dataflow().edges().size(), 2);
}

/**
 * Test state validation
 */
TEST_F(StateTest, StateValidation) {
    builder::SDFGBuilder builder("test_sdfg", FunctionType_CPU);

    auto& state = builder.add_state();
    auto& sdfg = builder.subject();

    // State should validate successfully (even if empty)
    EXPECT_NO_THROW(state.validate(sdfg));
}

/**
 * Test state symbol replacement
 */
TEST_F(StateTest, StateSymbolReplacement) {
    builder::SDFGBuilder builder("test_sdfg", FunctionType_CPU);

    builder.add_container("x", types::Scalar(types::PrimitiveType::Int32));

    auto& state = builder.add_state();
    auto& sdfg = builder.subject();

    symbolic::Symbol x = symbolic::symbol("x");
    symbolic::Symbol y = symbolic::symbol("y");

    // Replace x with y in state (affects dataflow graph)
    EXPECT_NO_THROW(state.replace(x, y));
}

/**
 * Test fixture for ReturnState tests
 */
class ReturnStateTest : public ::testing::Test {
};

/**
 * Test ReturnState with data container
 */
TEST_F(ReturnStateTest, ReturnStateWithData) {
    builder::SDFGBuilder builder("test_sdfg", FunctionType_CPU, types::Scalar(types::PrimitiveType::Double));

    builder.add_container("result", types::Scalar(types::PrimitiveType::Double));

    auto& return_state = builder.add_return_state("result");

    // Verify return state properties
    EXPECT_TRUE(return_state.is_data());
    EXPECT_FALSE(return_state.is_constant());
    EXPECT_EQ(return_state.data(), "result");
}

/**
 * Test ReturnState with constant value
 */
TEST_F(ReturnStateTest, ReturnStateWithConstant) {
    builder::SDFGBuilder builder("test_sdfg", FunctionType_CPU, types::Scalar(types::PrimitiveType::Int32));

    auto& return_state = builder.add_constant_return_state("42", types::Scalar(types::PrimitiveType::Int32));

    // Verify return state properties
    EXPECT_FALSE(return_state.is_data());
    EXPECT_TRUE(return_state.is_constant());
    EXPECT_EQ(return_state.data(), "42");
    EXPECT_EQ(return_state.type().type_id(), types::TypeID::Scalar);
}

/**
 * Test ReturnState with empty data (void return)
 */
TEST_F(ReturnStateTest, ReturnStateVoid) {
    builder::SDFGBuilder builder("test_sdfg", FunctionType_CPU);

    auto& return_state = builder.add_return_state("");

    // Verify return state properties
    EXPECT_TRUE(return_state.is_data());
    EXPECT_FALSE(return_state.is_constant());
    EXPECT_EQ(return_state.data(), "");
}

/**
 * Test that ReturnState cannot have outgoing edges
 */
TEST_F(ReturnStateTest, ReturnStateNoOutgoingEdges) {
    builder::SDFGBuilder builder("test_sdfg", FunctionType_CPU, types::Scalar(types::PrimitiveType::Double));

    builder.add_container("result", types::Scalar(types::PrimitiveType::Double));

    auto& return_state = builder.add_return_state("result");
    auto& sdfg = builder.subject();

    // Return state should have no outgoing edges
    EXPECT_EQ(sdfg.out_degree(return_state), 0);

    // Validation should succeed when there are no outgoing edges
    EXPECT_NO_THROW(return_state.validate(sdfg));
}

/**
 * Test that the builder prevents adding outgoing edges from ReturnState
 */
TEST_F(ReturnStateTest, ReturnStateBuilderPreventsOutgoingEdges) {
    builder::SDFGBuilder builder("test_sdfg", FunctionType_CPU, types::Scalar(types::PrimitiveType::Double));

    builder.add_container("result", types::Scalar(types::PrimitiveType::Double));

    auto& return_state = builder.add_return_state("result");
    auto& next_state = builder.add_state();

    // Try to add an outgoing edge from ReturnState - this should throw
    EXPECT_THROW(builder.add_edge(return_state, next_state), std::exception);
}

/**
 * Test ReturnState with invalid data (non-existent container)
 */
TEST_F(ReturnStateTest, ReturnStateInvalidData) {
    builder::SDFGBuilder builder("test_sdfg", FunctionType_CPU, types::Scalar(types::PrimitiveType::Double));

    // Return a container that doesn't exist
    auto& return_state = builder.add_return_state("nonexistent");
    auto& sdfg = builder.subject();

    // Validation should fail
    EXPECT_THROW(return_state.validate(sdfg), InvalidSDFGException);
}

/**
 * Test ReturnState after another state
 */
TEST_F(ReturnStateTest, ReturnStateAfterState) {
    builder::SDFGBuilder builder("test_sdfg", FunctionType_CPU, types::Scalar(types::PrimitiveType::Double));

    builder.add_container("result", types::Scalar(types::PrimitiveType::Double));

    auto& first_state = builder.add_state(true);
    auto& return_state = builder.add_return_state_after(first_state, "result");

    auto& sdfg = builder.subject();

    // Verify the edge between states
    EXPECT_EQ(sdfg.out_degree(first_state), 1);
    EXPECT_EQ(sdfg.in_degree(return_state), 1);
    EXPECT_TRUE(sdfg.is_adjacent(first_state, return_state));
}

/**
 * Test ReturnState symbol replacement
 */
TEST_F(ReturnStateTest, ReturnStateSymbolReplacement) {
    builder::SDFGBuilder builder("test_sdfg", FunctionType_CPU, types::Scalar(types::PrimitiveType::Int32));

    builder.add_container("x", types::Scalar(types::PrimitiveType::Int32));

    auto& return_state = builder.add_return_state("x");

    symbolic::Symbol x = symbolic::symbol("x");
    symbolic::Symbol y = symbolic::symbol("y");

    // Replace x with y in return state
    return_state.replace(x, y);

    // The data field should be updated
    EXPECT_EQ(return_state.data(), "y");
}

/**
 * Test multiple ReturnStates (different paths can return)
 */
TEST_F(ReturnStateTest, MultipleReturnStates) {
    builder::SDFGBuilder builder("test_sdfg", FunctionType_CPU, types::Scalar(types::PrimitiveType::Int32));

    builder.add_container("result1", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("result2", types::Scalar(types::PrimitiveType::Int32));

    auto& start_state = builder.add_state(true);
    auto& return_state1 = builder.add_return_state("result1");
    auto& return_state2 = builder.add_return_state("result2");

    builder.add_edge(start_state, return_state1);
    builder.add_edge(start_state, return_state2);

    auto& sdfg = builder.subject();

    // Both return states should be valid
    EXPECT_NO_THROW(return_state1.validate(sdfg));
    EXPECT_NO_THROW(return_state2.validate(sdfg));
}
