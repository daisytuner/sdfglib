#include "gtest/gtest.h"
#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/abs_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/add_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/maximum_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/minimum_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/exp_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/reduce_ops/max_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/reduce_ops/sum_node.h"

using namespace sdfg;

// Test integer support for operations that should support integers
TEST(TensorPrimitiveTypeTest, MaximumNodeInt32) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Int32);
    types::Pointer desc_ptr(desc);

    builder.add_container("a", desc_ptr);
    builder.add_container("b", desc_ptr);
    builder.add_container("c", desc_ptr);

    auto& block = builder.add_block(sdfg.root());
    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");
    auto& c_node = builder.add_access(block, "c");

    std::vector<symbolic::Expression> shape = {symbolic::integer(10)};
    auto& node = builder.add_library_node<math::tensor::MaximumNode>(block, DebugInfo(), shape);

    builder.add_computational_memlet(block, a_node, node, "A", {}, desc_ptr, block.debug_info());
    builder.add_computational_memlet(block, b_node, node, "B", {}, desc_ptr, block.debug_info());
    builder.add_computational_memlet(block, node, "C", c_node, {}, desc_ptr, block.debug_info());

    // Should not throw and should validate successfully
    EXPECT_NO_THROW(node.validate(*sdfg.root_function()));
    
    analysis::AnalysisManager analysis_manager(sdfg);
    EXPECT_TRUE(node.expand(builder, analysis_manager));
}

// Test float support for operations
TEST(TensorPrimitiveTypeTest, MaximumNodeFloat) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    builder.add_container("a", desc_ptr);
    builder.add_container("b", desc_ptr);
    builder.add_container("c", desc_ptr);

    auto& block = builder.add_block(sdfg.root());
    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");
    auto& c_node = builder.add_access(block, "c");

    std::vector<symbolic::Expression> shape = {symbolic::integer(10)};
    auto& node = builder.add_library_node<math::tensor::MaximumNode>(block, DebugInfo(), shape);

    builder.add_computational_memlet(block, a_node, node, "A", {}, desc_ptr, block.debug_info());
    builder.add_computational_memlet(block, b_node, node, "B", {}, desc_ptr, block.debug_info());
    builder.add_computational_memlet(block, node, "C", c_node, {}, desc_ptr, block.debug_info());

    EXPECT_NO_THROW(node.validate(*sdfg.root_function()));
    
    analysis::AnalysisManager analysis_manager(sdfg);
    EXPECT_TRUE(node.expand(builder, analysis_manager));
}

// Test double support for operations
TEST(TensorPrimitiveTypeTest, MaximumNodeDouble) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Double);
    types::Pointer desc_ptr(desc);

    builder.add_container("a", desc_ptr);
    builder.add_container("b", desc_ptr);
    builder.add_container("c", desc_ptr);

    auto& block = builder.add_block(sdfg.root());
    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");
    auto& c_node = builder.add_access(block, "c");

    std::vector<symbolic::Expression> shape = {symbolic::integer(10)};
    auto& node = builder.add_library_node<math::tensor::MaximumNode>(block, DebugInfo(), shape);

    builder.add_computational_memlet(block, a_node, node, "A", {}, desc_ptr, block.debug_info());
    builder.add_computational_memlet(block, b_node, node, "B", {}, desc_ptr, block.debug_info());
    builder.add_computational_memlet(block, node, "C", c_node, {}, desc_ptr, block.debug_info());

    EXPECT_NO_THROW(node.validate(*sdfg.root_function()));
    
    analysis::AnalysisManager analysis_manager(sdfg);
    EXPECT_TRUE(node.expand(builder, analysis_manager));
}

// Test unsigned integer support
TEST(TensorPrimitiveTypeTest, MaximumNodeUInt32) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::UInt32);
    types::Pointer desc_ptr(desc);

    builder.add_container("a", desc_ptr);
    builder.add_container("b", desc_ptr);
    builder.add_container("c", desc_ptr);

    auto& block = builder.add_block(sdfg.root());
    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");
    auto& c_node = builder.add_access(block, "c");

    std::vector<symbolic::Expression> shape = {symbolic::integer(10)};
    auto& node = builder.add_library_node<math::tensor::MaximumNode>(block, DebugInfo(), shape);

    builder.add_computational_memlet(block, a_node, node, "A", {}, desc_ptr, block.debug_info());
    builder.add_computational_memlet(block, b_node, node, "B", {}, desc_ptr, block.debug_info());
    builder.add_computational_memlet(block, node, "C", c_node, {}, desc_ptr, block.debug_info());

    EXPECT_NO_THROW(node.validate(*sdfg.root_function()));
    
    analysis::AnalysisManager analysis_manager(sdfg);
    EXPECT_TRUE(node.expand(builder, analysis_manager));
}

// Test that float-only operations reject integer types
TEST(TensorPrimitiveTypeTest, ExpNodeRejectsInt32) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Int32);
    types::Pointer desc_ptr(desc);

    builder.add_container("a", desc_ptr);
    builder.add_container("b", desc_ptr);

    auto& block = builder.add_block(sdfg.root());
    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");

    std::vector<symbolic::Expression> shape = {symbolic::integer(10)};
    auto& node = builder.add_library_node<math::tensor::ExpNode>(block, DebugInfo(), shape);

    builder.add_computational_memlet(block, a_node, node, "X", {}, desc_ptr, block.debug_info());
    builder.add_computational_memlet(block, node, "Y", b_node, {}, desc_ptr, block.debug_info());

    // Should throw because ExpNode doesn't support integer types
    EXPECT_THROW(node.validate(*sdfg.root_function()), InvalidSDFGException);
}

// Test that mixed types are rejected
TEST(TensorPrimitiveTypeTest, MixedTypesRejected) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar desc_float(types::PrimitiveType::Float);
    types::Scalar desc_double(types::PrimitiveType::Double);
    types::Pointer desc_float_ptr(desc_float);
    types::Pointer desc_double_ptr(desc_double);

    builder.add_container("a", desc_float_ptr);
    builder.add_container("b", desc_double_ptr);
    builder.add_container("c", desc_float_ptr);

    auto& block = builder.add_block(sdfg.root());
    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");
    auto& c_node = builder.add_access(block, "c");

    std::vector<symbolic::Expression> shape = {symbolic::integer(10)};
    auto& node = builder.add_library_node<math::tensor::AddNode>(block, DebugInfo(), shape);

    builder.add_computational_memlet(block, a_node, node, "A", {}, desc_float_ptr, block.debug_info());
    builder.add_computational_memlet(block, b_node, node, "B", {}, desc_double_ptr, block.debug_info());
    builder.add_computational_memlet(block, node, "C", c_node, {}, desc_float_ptr, block.debug_info());

    // Should throw because types are mixed (Float and Double)
    EXPECT_THROW(node.validate(*sdfg.root_function()), InvalidSDFGException);
}

// Test AbsNode with integers
TEST(TensorPrimitiveTypeTest, AbsNodeInt32) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Int32);
    types::Pointer desc_ptr(desc);

    builder.add_container("a", desc_ptr);
    builder.add_container("b", desc_ptr);

    auto& block = builder.add_block(sdfg.root());
    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");

    std::vector<symbolic::Expression> shape = {symbolic::integer(10)};
    auto& node = builder.add_library_node<math::tensor::AbsNode>(block, DebugInfo(), shape);

    builder.add_computational_memlet(block, a_node, node, "X", {}, desc_ptr, block.debug_info());
    builder.add_computational_memlet(block, node, "Y", b_node, {}, desc_ptr, block.debug_info());

    EXPECT_NO_THROW(node.validate(*sdfg.root_function()));
    
    analysis::AnalysisManager analysis_manager(sdfg);
    EXPECT_TRUE(node.expand(builder, analysis_manager));
}

// Test reduce max with integers
TEST(TensorPrimitiveTypeTest, ReduceMaxInt64) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Int64);
    types::Pointer desc_ptr(desc);

    builder.add_container("a", desc_ptr);
    builder.add_container("b", desc_ptr);

    auto& block = builder.add_block(sdfg.root());
    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");

    std::vector<symbolic::Expression> shape = {symbolic::integer(10), symbolic::integer(20)};
    std::vector<int64_t> axes = {1};
    auto& node = builder.add_library_node<math::tensor::MaxNode>(block, DebugInfo(), shape, axes, false);

    builder.add_computational_memlet(block, a_node, node, "X", {}, desc_ptr, block.debug_info());
    builder.add_computational_memlet(block, node, "Y", b_node, {}, desc_ptr, block.debug_info());

    EXPECT_NO_THROW(node.validate(*sdfg.root_function()));
    
    analysis::AnalysisManager analysis_manager(sdfg);
    EXPECT_TRUE(node.expand(builder, analysis_manager));
}

// Test reduce sum with integers
TEST(TensorPrimitiveTypeTest, ReduceSumInt32) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Int32);
    types::Pointer desc_ptr(desc);

    builder.add_container("a", desc_ptr);
    builder.add_container("b", desc_ptr);

    auto& block = builder.add_block(sdfg.root());
    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");

    std::vector<symbolic::Expression> shape = {symbolic::integer(10), symbolic::integer(20)};
    std::vector<int64_t> axes = {0};
    auto& node = builder.add_library_node<math::tensor::SumNode>(block, DebugInfo(), shape, axes, false);

    builder.add_computational_memlet(block, a_node, node, "X", {}, desc_ptr, block.debug_info());
    builder.add_computational_memlet(block, node, "Y", b_node, {}, desc_ptr, block.debug_info());

    EXPECT_NO_THROW(node.validate(*sdfg.root_function()));
    
    analysis::AnalysisManager analysis_manager(sdfg);
    EXPECT_TRUE(node.expand(builder, analysis_manager));
}

// Test with scalar inputs (not pointers)
TEST(TensorPrimitiveTypeTest, ScalarInputFloat) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar desc_scalar(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc_scalar);

    builder.add_container("a", desc_scalar);
    builder.add_container("b", desc_ptr);
    builder.add_container("c", desc_ptr);

    auto& block = builder.add_block(sdfg.root());
    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");
    auto& c_node = builder.add_access(block, "c");

    std::vector<symbolic::Expression> shape = {symbolic::integer(10)};
    auto& node = builder.add_library_node<math::tensor::AddNode>(block, DebugInfo(), shape);

    // Connect scalar and pointer - should still validate
    builder.add_computational_memlet(block, a_node, node, "A", {}, desc_scalar, block.debug_info());
    builder.add_computational_memlet(block, b_node, node, "B", {}, desc_ptr, block.debug_info());
    builder.add_computational_memlet(block, node, "C", c_node, {}, desc_ptr, block.debug_info());

    EXPECT_NO_THROW(node.validate(*sdfg.root_function()));
}
