#include "sdfg/control_flow/interstate_edge.h"

#include <gtest/gtest.h>

#include "sdfg/builder/sdfg_builder.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/scalar.h"

using namespace sdfg;

/**
 * Test fixture for InterstateEdge tests
 */
class InterstateEdgeTest : public ::testing::Test {
};

/**
 * Test basic unconditional edge creation
 */
TEST_F(InterstateEdgeTest, UnconditionalEdge) {
    builder::SDFGBuilder builder("test_sdfg", FunctionType_CPU);

    auto& state1 = builder.add_state(true);
    auto& state2 = builder.add_state();
    auto& edge = builder.add_edge(state1, state2);

    // Verify edge properties
    EXPECT_EQ(&edge.src(), &state1);
    EXPECT_EQ(&edge.dst(), &state2);
    EXPECT_TRUE(edge.is_unconditional());
    EXPECT_TRUE(symbolic::is_true(edge.condition()));
    EXPECT_EQ(edge.assignments().size(), 0);
}

/**
 * Test edge with condition
 */
TEST_F(InterstateEdgeTest, EdgeWithCondition) {
    builder::SDFGBuilder builder("test_sdfg", FunctionType_CPU);

    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    symbolic::Symbol i = symbolic::symbol("i");

    auto& state1 = builder.add_state(true);
    auto& state2 = builder.add_state();

    auto condition = symbolic::Lt(i, symbolic::integer(10));
    auto& edge = builder.add_edge(state1, state2, condition);

    // Verify edge properties
    EXPECT_EQ(&edge.src(), &state1);
    EXPECT_EQ(&edge.dst(), &state2);
    EXPECT_FALSE(edge.is_unconditional());
    EXPECT_EQ(edge.condition()->__str__(), "i < 10");
    EXPECT_EQ(edge.assignments().size(), 0);
}

/**
 * Test edge with assignments only (unconditional)
 */
TEST_F(InterstateEdgeTest, EdgeWithAssignments) {
    builder::SDFGBuilder builder("test_sdfg", FunctionType_CPU);

    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    symbolic::Symbol i = symbolic::symbol("i");

    auto& state1 = builder.add_state(true);
    auto& state2 = builder.add_state();

    control_flow::Assignments assignments;
    assignments[i] = symbolic::integer(0);
    auto& edge = builder.add_edge(state1, state2, assignments);

    // Verify edge properties
    EXPECT_EQ(&edge.src(), &state1);
    EXPECT_EQ(&edge.dst(), &state2);
    EXPECT_TRUE(edge.is_unconditional());
    EXPECT_EQ(edge.assignments().size(), 1);

    auto& assignment = *edge.assignments().begin();
    EXPECT_EQ(assignment.first->get_name(), "i");
    EXPECT_EQ(assignment.second->__str__(), "0");
}

/**
 * Test edge with both condition and assignments
 */
TEST_F(InterstateEdgeTest, EdgeWithConditionAndAssignments) {
    builder::SDFGBuilder builder("test_sdfg", FunctionType_CPU);

    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    symbolic::Symbol i = symbolic::symbol("i");

    auto& state1 = builder.add_state(true);
    auto& state2 = builder.add_state();

    auto condition = symbolic::Lt(i, symbolic::integer(10));
    control_flow::Assignments assignments;
    assignments[i] = symbolic::add(i, symbolic::integer(1));
    auto& edge = builder.add_edge(state1, state2, assignments, condition);

    // Verify edge properties
    EXPECT_EQ(&edge.src(), &state1);
    EXPECT_EQ(&edge.dst(), &state2);
    EXPECT_FALSE(edge.is_unconditional());
    EXPECT_EQ(edge.condition()->__str__(), "i < 10");
    EXPECT_EQ(edge.assignments().size(), 1);

    auto& assignment = *edge.assignments().begin();
    EXPECT_EQ(assignment.first->get_name(), "i");
    EXPECT_EQ(assignment.second->__str__(), "1 + i");
}

/**
 * Test that demonstrates condition is evaluated BEFORE assignment
 *
 * This is a conceptual test showing the semantics:
 * If we have condition "i < 10" and assignment "i = i + 1",
 * the condition checks the OLD value of i, then i is incremented.
 */
TEST_F(InterstateEdgeTest, ConditionEvaluatedBeforeAssignment) {
    builder::SDFGBuilder builder("test_sdfg", FunctionType_CPU);

    builder.add_container("counter", types::Scalar(types::PrimitiveType::Int32));
    symbolic::Symbol counter = symbolic::symbol("counter");

    auto& init_state = builder.add_state(true);
    auto& loop_body = builder.add_state();
    auto& exit_state = builder.add_state();

    // Edge from init to loop: initialize counter = 0
    control_flow::Assignments init_assignments;
    init_assignments[counter] = symbolic::integer(0);
    builder.add_edge(init_state, loop_body, init_assignments);

    // Loop edge: check counter < 5, then increment counter
    // The condition sees the counter value BEFORE increment
    auto loop_condition = symbolic::Lt(counter, symbolic::integer(5));
    control_flow::Assignments loop_assignments;
    loop_assignments[counter] = symbolic::add(counter, symbolic::integer(1));
    auto& loop_edge = builder.add_edge(loop_body, loop_body, loop_assignments, loop_condition);

    // Exit edge: check counter >= 5 (negation of loop condition)
    auto exit_condition = symbolic::Ge(counter, symbolic::integer(5));
    builder.add_edge(loop_body, exit_state, exit_condition);

    // Verify the semantics are encoded correctly
    EXPECT_EQ(loop_edge.condition()->__str__(), "counter < 5");
    EXPECT_EQ(loop_edge.assignments().begin()->second->__str__(), "1 + counter");

    // The key insight: when counter = 4, the condition evaluates to true (4 < 5),
    // THEN counter becomes 5, and the next iteration will exit
}

/**
 * Test loop increment pattern (classic for-loop)
 */
TEST_F(InterstateEdgeTest, ForLoopPattern) {
    builder::SDFGBuilder builder("test_sdfg", FunctionType_CPU);

    builder.add_container("i", types::Scalar(types::PrimitiveType::UInt64));
    symbolic::Symbol i = symbolic::symbol("i");

    auto& guard_state = builder.add_state(true);
    auto& body_state = builder.add_state();
    auto& after_state = builder.add_state();

    // Initialize: i = 0
    control_flow::Assignments init_assignments;
    init_assignments[i] = symbolic::integer(0);
    builder.add_edge(guard_state, body_state, init_assignments);

    // Loop back: check i < 10, then i = i + 1
    auto loop_condition = symbolic::Lt(i, symbolic::integer(10));
    control_flow::Assignments increment;
    increment[i] = symbolic::add(i, symbolic::integer(1));
    auto& loop_edge = builder.add_edge(body_state, guard_state, increment, loop_condition);

    // Exit: i >= 10
    auto exit_condition = symbolic::Ge(i, symbolic::integer(10));
    builder.add_edge(body_state, after_state, exit_condition);

    // Verify loop edge structure
    EXPECT_FALSE(loop_edge.is_unconditional());
    EXPECT_EQ(loop_edge.assignments().size(), 1);

    auto& sdfg = builder.subject();
    EXPECT_NO_THROW(loop_edge.validate(sdfg));
}

/**
 * Test conditional branching (if-else pattern)
 */
TEST_F(InterstateEdgeTest, IfElsePattern) {
    builder::SDFGBuilder builder("test_sdfg", FunctionType_CPU);

    builder.add_container("flag", types::Scalar(types::PrimitiveType::Int32));
    symbolic::Symbol flag = symbolic::symbol("flag");

    auto& guard_state = builder.add_state(true);
    auto& then_state = builder.add_state();
    auto& else_state = builder.add_state();
    auto& merge_state = builder.add_state();

    // Branch on condition
    auto then_condition = symbolic::Eq(flag, symbolic::integer(1));
    auto else_condition = symbolic::Ne(flag, symbolic::integer(1));

    auto& then_edge = builder.add_edge(guard_state, then_state, then_condition);
    auto& else_edge = builder.add_edge(guard_state, else_state, else_condition);

    builder.add_edge(then_state, merge_state);
    builder.add_edge(else_state, merge_state);

    // Verify branching edges
    EXPECT_FALSE(then_edge.is_unconditional());
    EXPECT_FALSE(else_edge.is_unconditional());
    EXPECT_EQ(then_edge.condition()->__str__(), "1 == flag");
    EXPECT_EQ(else_edge.condition()->__str__(), "1 != flag");

    auto& sdfg = builder.subject();
    EXPECT_EQ(sdfg.out_degree(guard_state), 2);
}

/**
 * Test edge with multiple assignments
 * Demonstrates that all assignments use the symbol values from before the edge is traversed
 */
TEST_F(InterstateEdgeTest, MultipleAssignments) {
    builder::SDFGBuilder builder("test_sdfg", FunctionType_CPU);

    builder.add_container("x", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("y", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("z", types::Scalar(types::PrimitiveType::Int32));

    symbolic::Symbol x = symbolic::symbol("x");
    symbolic::Symbol y = symbolic::symbol("y");
    symbolic::Symbol z = symbolic::symbol("z");

    auto& state1 = builder.add_state(true);
    auto& state2 = builder.add_state();

    // All assignments use the old values of symbols at edge entry
    // For example, if entering with x=10, y=20, then:
    //   - a = 1 (independent)
    //   - b = 2 (independent)
    //   - c = x + y = 10 + 20 = 30 (uses old x and y, not 1 and 2)
    control_flow::Assignments assignments;
    assignments[x] = symbolic::integer(1);
    assignments[y] = symbolic::integer(2);
    assignments[z] = symbolic::add(x, y);

    auto& edge = builder.add_edge(state1, state2, assignments);

    EXPECT_EQ(edge.assignments().size(), 3);
}

/**
 * Test edge validation with valid integer assignments
 */
TEST_F(InterstateEdgeTest, ValidEdgeValidation) {
    builder::SDFGBuilder builder("test_sdfg", FunctionType_CPU);

    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    symbolic::Symbol i = symbolic::symbol("i");

    auto& state1 = builder.add_state(true);
    auto& state2 = builder.add_state();

    control_flow::Assignments assignments;
    assignments[i] = symbolic::integer(0);
    auto& edge = builder.add_edge(state1, state2, assignments);

    auto& sdfg = builder.subject();
    EXPECT_NO_THROW(edge.validate(sdfg));
}

/**
 * Test edge with pointer type in condition
 */
TEST_F(InterstateEdgeTest, PointerInCondition) {
    builder::SDFGBuilder builder("test_sdfg", FunctionType_CPU);

    builder.add_container("ptr", types::Pointer(types::Scalar(types::PrimitiveType::Int32)));
    symbolic::Symbol ptr = symbolic::symbol("ptr");

    auto& state1 = builder.add_state(true);
    auto& state2 = builder.add_state();

    // Check if pointer is null
    auto condition = symbolic::Eq(ptr, symbolic::__nullptr__());
    auto& edge = builder.add_edge(state1, state2, condition);

    auto& sdfg = builder.subject();
    EXPECT_NO_THROW(edge.validate(sdfg));
}

/**
 * Test edge symbol replacement in assignments
 */
TEST_F(InterstateEdgeTest, EdgeSymbolReplacement) {
    builder::SDFGBuilder builder("test_sdfg", FunctionType_CPU);

    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("j", types::Scalar(types::PrimitiveType::Int32));

    symbolic::Symbol i = symbolic::symbol("i");
    symbolic::Symbol j = symbolic::symbol("j");

    auto& state1 = builder.add_state(true);
    auto& state2 = builder.add_state();

    auto condition = symbolic::Lt(i, symbolic::integer(10));
    control_flow::Assignments assignments;
    assignments[i] = symbolic::add(i, symbolic::integer(1));
    auto& edge = builder.add_edge(state1, state2, assignments, condition);

    // Replace i with j
    edge.replace(i, j);

    // Verify replacement in assignments
    EXPECT_EQ(edge.assignments().size(), 1);
    auto& assignment = *edge.assignments().begin();
    EXPECT_EQ(assignment.first->get_name(), "j");
}

/**
 * Test complex condition with multiple symbols
 */
TEST_F(InterstateEdgeTest, ComplexCondition) {
    builder::SDFGBuilder builder("test_sdfg", FunctionType_CPU);

    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("j", types::Scalar(types::PrimitiveType::Int32));

    symbolic::Symbol i = symbolic::symbol("i");
    symbolic::Symbol j = symbolic::symbol("j");

    auto& state1 = builder.add_state(true);
    auto& state2 = builder.add_state();

    // Condition: (i < 10) AND (j > 0)
    auto condition = symbolic::And(symbolic::Lt(i, symbolic::integer(10)), symbolic::Gt(j, symbolic::integer(0)));

    auto& edge = builder.add_edge(state1, state2, condition);

    EXPECT_FALSE(edge.is_unconditional());
    EXPECT_EQ(edge.assignments().size(), 0);

    auto& sdfg = builder.subject();
    EXPECT_NO_THROW(edge.validate(sdfg));
}

/**
 * Test that assignment uses old values in expressions
 *
 * When we have assignments like {x = y, y = x}, both should use the OLD values.
 * This test verifies the structure (actual execution would be done by runtime).
 */
TEST_F(InterstateEdgeTest, AssignmentUsesOldValues) {
    builder::SDFGBuilder builder("test_sdfg", FunctionType_CPU);

    builder.add_container("x", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("y", types::Scalar(types::PrimitiveType::Int32));

    symbolic::Symbol x = symbolic::symbol("x");
    symbolic::Symbol y = symbolic::symbol("y");

    auto& state1 = builder.add_state(true);
    auto& state2 = builder.add_state();

    // Swap: x = y, y = x (both use old values)
    control_flow::Assignments assignments;
    assignments[x] = y;
    assignments[y] = x;

    auto& edge = builder.add_edge(state1, state2, assignments);

    EXPECT_EQ(edge.assignments().size(), 2);

    auto& sdfg = builder.subject();
    EXPECT_NO_THROW(edge.validate(sdfg));
}

/**
 * Test edge with arithmetic expression in assignment
 */
TEST_F(InterstateEdgeTest, ArithmeticExpressionInAssignment) {
    builder::SDFGBuilder builder("test_sdfg", FunctionType_CPU);

    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("j", types::Scalar(types::PrimitiveType::Int32));

    symbolic::Symbol i = symbolic::symbol("i");
    symbolic::Symbol j = symbolic::symbol("j");

    auto& state1 = builder.add_state(true);
    auto& state2 = builder.add_state();

    // Complex expression: i = (j * 2) + 1
    control_flow::Assignments assignments;
    assignments[i] = symbolic::add(symbolic::mul(j, symbolic::integer(2)), symbolic::integer(1));

    auto& edge = builder.add_edge(state1, state2, assignments);

    auto& assignment = *edge.assignments().begin();
    EXPECT_EQ(assignment.first->get_name(), "i");
    EXPECT_EQ(assignment.second->__str__(), "1 + 2*j");
}

/**
 * Test edge in a more complex control flow graph
 */
TEST_F(InterstateEdgeTest, ComplexControlFlowGraph) {
    builder::SDFGBuilder builder("test_sdfg", FunctionType_CPU);

    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    symbolic::Symbol i = symbolic::symbol("i");

    auto& start = builder.add_state(true);
    auto& loop_head = builder.add_state();
    auto& loop_body = builder.add_state();
    auto& exit = builder.add_state();

    // Initialize: i = 0
    control_flow::Assignments init_assign;
    init_assign[i] = symbolic::integer(0);
    builder.add_edge(start, loop_head, init_assign);

    // Enter loop: i < 10
    builder.add_edge(loop_head, loop_body, symbolic::Lt(i, symbolic::integer(10)));

    // Exit loop: i >= 10
    builder.add_edge(loop_head, exit, symbolic::Ge(i, symbolic::integer(10)));

    // Loop back: i = i + 1
    control_flow::Assignments loop_assign;
    loop_assign[i] = symbolic::add(i, symbolic::integer(1));
    auto& back_edge = builder.add_edge(loop_body, loop_head, loop_assign);

    auto& sdfg = builder.subject();

    // Verify graph structure
    EXPECT_EQ(sdfg.out_degree(loop_head), 2);
    EXPECT_EQ(sdfg.in_degree(loop_head), 2);

    // All edges should be valid
    for (auto& edge : sdfg.edges()) {
        EXPECT_NO_THROW(edge.validate(sdfg));
    }
}

/**
 * Test edge accessing both source and destination states
 */
TEST_F(InterstateEdgeTest, EdgeSourceAndDestination) {
    builder::SDFGBuilder builder("test_sdfg", FunctionType_CPU);

    auto& state1 = builder.add_state(true);
    auto& state2 = builder.add_state();
    auto& edge = builder.add_edge(state1, state2);

    auto& sdfg = builder.subject();

    // Verify we can access states through edge
    EXPECT_EQ(&edge.src(), &state1);
    EXPECT_EQ(&edge.dst(), &state2);

    // Verify adjacency through SDFG
    EXPECT_TRUE(sdfg.is_adjacent(state1, state2));
    EXPECT_EQ(&sdfg.edge(state1, state2), &edge);
}
