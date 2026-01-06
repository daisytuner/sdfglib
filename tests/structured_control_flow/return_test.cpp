#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/return.h"
#include "sdfg/types/scalar.h"
#include "sdfg/symbolic/symbolic.h"

using namespace sdfg;

namespace sdfg::structured_control_flow {

// Test basic Return structure and pointers
TEST(ReturnTest, BasicStructure) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU, types::Scalar(types::PrimitiveType::Int32));
    
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("x", int_type);
    
    auto& root = builder.subject().root();
    
    auto& return_node = builder.add_return(root, "x");
    
    // Verify return_node is a ControlFlowNode
    EXPECT_TRUE(dynamic_cast<const ControlFlowNode*>(&return_node) != nullptr);
    
    // Verify return_node is a Return
    EXPECT_TRUE(dynamic_cast<const Return*>(&return_node) != nullptr);
}

// Test data return
TEST(ReturnTest, DataReturn) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU, types::Scalar(types::PrimitiveType::Int32));
    
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("result", int_type);
    
    auto& root = builder.subject().root();
    
    auto& return_node = builder.add_return(root, "result");
    
    // Verify it's a data return
    EXPECT_TRUE(return_node.is_data());
    EXPECT_FALSE(return_node.is_constant());
    EXPECT_EQ(return_node.data(), "result");
}

// Test constant return
TEST(ReturnTest, ConstantReturn) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU, types::Scalar(types::PrimitiveType::Int32));
    
    auto& root = builder.subject().root();
    
    types::Scalar int_type(types::PrimitiveType::Int32);
    auto& return_node = builder.add_return(root, "42", int_type);
    
    // Verify it's a constant return
    EXPECT_TRUE(return_node.is_constant());
    EXPECT_FALSE(return_node.is_data());
    EXPECT_EQ(return_node.data(), "42");
}

// Test return type
TEST(ReturnTest, ReturnType) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU, types::Scalar(types::PrimitiveType::Float32));
    
    types::Scalar float_type(types::PrimitiveType::Float32);
    builder.add_container("value", float_type);
    
    auto& root = builder.subject().root();
    
    auto& return_node = builder.add_return(root, "value");
    
    // Verify type
    const auto& ret_type = return_node.type();
    EXPECT_TRUE(dynamic_cast<const types::Scalar*>(&ret_type) != nullptr);
    
    const auto& scalar_type = dynamic_cast<const types::Scalar&>(ret_type);
    EXPECT_EQ(scalar_type.type().value(), types::PrimitiveType::Float32.value());
}

// Test return with different types
TEST(ReturnTest, DifferentTypes) {
    // Int32
    {
        builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU, types::Scalar(types::PrimitiveType::Int32));
        types::Scalar int_type(types::PrimitiveType::Int32);
        auto& root = builder.subject().root();
        auto& return_node = builder.add_return(root, "123", int_type);
        EXPECT_TRUE(return_node.is_constant());
    }
    
    // Float64
    {
        builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU, types::Scalar(types::PrimitiveType::Float64));
        types::Scalar float_type(types::PrimitiveType::Float64);
        auto& root = builder.subject().root();
        auto& return_node = builder.add_return(root, "3.14", float_type);
        EXPECT_TRUE(return_node.is_constant());
    }
    
    // Bool
    {
        builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU, types::Scalar(types::PrimitiveType::Bool));
        types::Scalar bool_type(types::PrimitiveType::Bool);
        auto& root = builder.subject().root();
        auto& return_node = builder.add_return(root, "true", bool_type);
        EXPECT_TRUE(return_node.is_constant());
    }
}

// Test return in conditional
TEST(ReturnTest, ReturnInConditional) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU, types::Scalar(types::PrimitiveType::Int32));
    
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("x", int_type);
    builder.add_container("y", int_type);
    
    auto& root = builder.subject().root();
    
    auto& if_else = builder.add_if_else(root);
    
    // Return x if condition is true
    auto& if_seq = builder.add_case(if_else, symbolic::Gt(symbolic::symbol("x"), symbolic::integer(0)));
    auto& return_x = builder.add_return(if_seq, "x");
    EXPECT_TRUE(return_x.is_data());
    EXPECT_EQ(return_x.data(), "x");
    
    // Return y otherwise
    auto& else_seq = builder.add_case(if_else, symbolic::Le(symbolic::symbol("x"), symbolic::integer(0)));
    auto& return_y = builder.add_return(else_seq, "y");
    EXPECT_TRUE(return_y.is_data());
    EXPECT_EQ(return_y.data(), "y");
}

// Test return at end of sequence
TEST(ReturnTest, ReturnAtEndOfSequence) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU, types::Scalar(types::PrimitiveType::Int32));
    
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("result", int_type);
    
    auto& root = builder.subject().root();
    
    // Add some blocks
    builder.add_block(root, control_flow::Assignments{});
    builder.add_block(root, control_flow::Assignments{});
    
    // Add return at the end
    auto& return_node = builder.add_return(root, "result");
    
    EXPECT_EQ(root.size(), 3);
    
    // Verify last element is the return
    auto [node, transition] = root.at(2);
    EXPECT_TRUE(dynamic_cast<const Return*>(&node) != nullptr);
}

// Test multiple return paths
TEST(ReturnTest, MultipleReturnPaths) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU, types::Scalar(types::PrimitiveType::Int32));
    
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("x", int_type);
    
    auto& root = builder.subject().root();
    auto& if_else = builder.add_if_else(root);
    
    // First case
    auto& seq1 = builder.add_case(if_else, symbolic::Gt(symbolic::symbol("x"), symbolic::integer(10)));
    builder.add_return(seq1, "100", int_type);
    
    // Second case
    auto& seq2 = builder.add_case(if_else, symbolic::And(
        symbolic::Gt(symbolic::symbol("x"), symbolic::integer(0)),
        symbolic::Le(symbolic::symbol("x"), symbolic::integer(10))
    ));
    builder.add_return(seq2, "50", int_type);
    
    // Third case (else)
    auto& seq3 = builder.add_case(if_else, symbolic::Le(symbolic::symbol("x"), symbolic::integer(0)));
    builder.add_return(seq3, "0", int_type);
    
    // All three branches have returns
    EXPECT_EQ(seq1.size(), 1);
    EXPECT_EQ(seq2.size(), 1);
    EXPECT_EQ(seq3.size(), 1);
}

} // namespace sdfg::structured_control_flow
