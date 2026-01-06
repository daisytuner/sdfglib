#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/symbolic/symbolic.h"

using namespace sdfg;

namespace sdfg::structured_control_flow {

// Test basic Sequence structure and pointers
TEST(SequenceTest, BasicStructure) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    
    auto& root = builder.subject().root();
    
    // Root is a Sequence
    EXPECT_TRUE(dynamic_cast<const Sequence*>(&root) != nullptr);
    EXPECT_EQ(root.size(), 0);
}

// Test children and transitions
TEST(SequenceTest, ChildrenAndTransitions) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("x", int_type);
    
    auto& root = builder.subject().root();
    
    // Add blocks with transitions
    builder.add_block(root, control_flow::Assignments{
        {symbolic::symbol("x"), symbolic::integer(1)}
    });
    builder.add_block(root, control_flow::Assignments{
        {symbolic::symbol("x"), symbolic::integer(2)}
    });
    builder.add_block(root, control_flow::Assignments{
        {symbolic::symbol("x"), symbolic::integer(3)}
    });
    
    EXPECT_EQ(root.size(), 3);
    
    // Verify each child has a transition
    for (size_t i = 0; i < root.size(); ++i) {
        auto [node, transition] = root.at(i);
        EXPECT_TRUE(dynamic_cast<const ControlFlowNode*>(&node) != nullptr);
        EXPECT_TRUE(dynamic_cast<const Transition*>(&transition) != nullptr);
    }
}

// Test at() method for accessing children
TEST(SequenceTest, AtMethod) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    
    auto& root = builder.subject().root();
    
    builder.add_block(root, control_flow::Assignments{});
    builder.add_block(root, control_flow::Assignments{});
    
    // Test const version
    const auto& const_root = root;
    auto [const_node1, const_trans1] = const_root.at(0);
    auto [const_node2, const_trans2] = const_root.at(1);
    
    EXPECT_TRUE(dynamic_cast<const Block*>(&const_node1) != nullptr);
    EXPECT_TRUE(dynamic_cast<const Block*>(&const_node2) != nullptr);
    
    // Test non-const version
    auto [mut_node1, mut_trans1] = root.at(0);
    auto [mut_node2, mut_trans2] = root.at(1);
    
    EXPECT_TRUE(dynamic_cast<Block*>(&mut_node1) != nullptr);
    EXPECT_TRUE(dynamic_cast<Block*>(&mut_node2) != nullptr);
}

// Test transition assignments
TEST(SequenceTest, TransitionAssignments) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("x", int_type);
    builder.add_container("y", int_type);
    
    auto& root = builder.subject().root();
    
    control_flow::Assignments assignments1{
        {symbolic::symbol("x"), symbolic::integer(10)}
    };
    control_flow::Assignments assignments2{
        {symbolic::symbol("x"), symbolic::integer(20)},
        {symbolic::symbol("y"), symbolic::integer(30)}
    };
    
    builder.add_block(root, assignments1);
    builder.add_block(root, assignments2);
    
    // Verify assignments
    auto [node1, trans1] = root.at(0);
    auto [node2, trans2] = root.at(1);
    
    EXPECT_EQ(trans1.assignments().size(), 1);
    EXPECT_TRUE(symbolic::eq(
        trans1.assignments().at(symbolic::symbol("x")),
        symbolic::integer(10)
    ));
    
    EXPECT_EQ(trans2.assignments().size(), 2);
    EXPECT_TRUE(symbolic::eq(
        trans2.assignments().at(symbolic::symbol("x")),
        symbolic::integer(20)
    ));
    EXPECT_TRUE(symbolic::eq(
        trans2.assignments().at(symbolic::symbol("y")),
        symbolic::integer(30)
    ));
}

// Test empty transition
TEST(SequenceTest, EmptyTransition) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    
    auto& root = builder.subject().root();
    
    builder.add_block(root, control_flow::Assignments{});
    
    auto [node, transition] = root.at(0);
    
    EXPECT_TRUE(transition.empty());
    EXPECT_EQ(transition.size(), 0);
}

// Test non-empty transition
TEST(SequenceTest, NonEmptyTransition) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("x", int_type);
    
    auto& root = builder.subject().root();
    
    builder.add_block(root, control_flow::Assignments{
        {symbolic::symbol("x"), symbolic::integer(5)}
    });
    
    auto [node, transition] = root.at(0);
    
    EXPECT_FALSE(transition.empty());
    EXPECT_EQ(transition.size(), 1);
}

// Test transition parent
TEST(SequenceTest, TransitionParent) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    
    auto& root = builder.subject().root();
    
    builder.add_block(root, control_flow::Assignments{});
    
    auto [node, transition] = root.at(0);
    
    // Verify transition knows its parent
    EXPECT_EQ(&transition.parent(), &root);
    
    // Const version
    const auto& const_transition = transition;
    EXPECT_EQ(&const_transition.parent(), &root);
}

// Test nested sequences
TEST(SequenceTest, NestedSequences) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("x", int_type);
    
    auto& root = builder.subject().root();
    
    // Add if-else which contains nested sequences
    auto& if_else = builder.add_if_else(root);
    auto& if_seq = builder.add_case(if_else, symbolic::Gt(symbolic::symbol("x"), symbolic::integer(0)));
    auto& else_seq = builder.add_case(if_else, symbolic::Le(symbolic::symbol("x"), symbolic::integer(0)));
    
    // Both are sequences
    EXPECT_TRUE(dynamic_cast<const Sequence*>(&if_seq) != nullptr);
    EXPECT_TRUE(dynamic_cast<const Sequence*>(&else_seq) != nullptr);
    
    // Add blocks to nested sequences
    builder.add_block(if_seq, control_flow::Assignments{});
    builder.add_block(else_seq, control_flow::Assignments{});
    
    EXPECT_EQ(if_seq.size(), 1);
    EXPECT_EQ(else_seq.size(), 1);
}

} // namespace sdfg::structured_control_flow
