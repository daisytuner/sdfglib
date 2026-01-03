/**
 * @file mlir_frontend_test.cpp
 * @brief Tests for MLIR frontend type conversions and operation mappings
 */

#include <gtest/gtest.h>

#include "sdfg/frontend/mlir_frontend.h"

using namespace sdfg;
using namespace sdfg::frontend;

/**
 * Test scalar type conversions from MLIR to SDFG
 */
TEST(MLIRFrontendTest, ScalarTypeConversion) {
    MLIRFrontend frontend;

    // Test integer types
    auto i1_type = frontend.convert_scalar_type("i1");
    EXPECT_EQ(i1_type.primitive_type(), types::PrimitiveType::Bool);

    auto i8_type = frontend.convert_scalar_type("i8");
    EXPECT_EQ(i8_type.primitive_type(), types::PrimitiveType::Int8);

    auto i16_type = frontend.convert_scalar_type("i16");
    EXPECT_EQ(i16_type.primitive_type(), types::PrimitiveType::Int16);

    auto i32_type = frontend.convert_scalar_type("i32");
    EXPECT_EQ(i32_type.primitive_type(), types::PrimitiveType::Int32);

    auto i64_type = frontend.convert_scalar_type("i64");
    EXPECT_EQ(i64_type.primitive_type(), types::PrimitiveType::Int64);

    // Test floating point types
    auto f16_type = frontend.convert_scalar_type("f16");
    EXPECT_EQ(f16_type.primitive_type(), types::PrimitiveType::Half);

    auto f32_type = frontend.convert_scalar_type("f32");
    EXPECT_EQ(f32_type.primitive_type(), types::PrimitiveType::Float);

    auto f64_type = frontend.convert_scalar_type("f64");
    EXPECT_EQ(f64_type.primitive_type(), types::PrimitiveType::Double);

    // Test index type (maps to i64)
    auto index_type = frontend.convert_scalar_type("index");
    EXPECT_EQ(index_type.primitive_type(), types::PrimitiveType::Int64);
}

/**
 * Test unsupported scalar type throws exception
 */
TEST(MLIRFrontendTest, UnsupportedScalarType) {
    MLIRFrontend frontend;

    EXPECT_THROW(frontend.convert_scalar_type("unknown_type"), std::invalid_argument);
    EXPECT_THROW(frontend.convert_scalar_type("i128"), std::invalid_argument);
}

/**
 * Test tensor type conversion to flat pointer
 */
TEST(MLIRFrontendTest, TensorTypeConversion) {
    MLIRFrontend frontend;

    std::vector<int64_t> shape = {32, 64};

    // Test f32 tensor
    auto f32_tensor = frontend.convert_tensor_type("f32", shape);
    EXPECT_EQ(f32_tensor.type_id(), types::TypeID::Pointer);
    EXPECT_TRUE(f32_tensor.has_pointee_type());
    auto& pointee = f32_tensor.pointee_type();
    EXPECT_EQ(pointee.type_id(), types::TypeID::Scalar);
    EXPECT_EQ(pointee.primitive_type(), types::PrimitiveType::Float);

    // Test i32 tensor
    auto i32_tensor = frontend.convert_tensor_type("i32", shape);
    EXPECT_EQ(i32_tensor.type_id(), types::TypeID::Pointer);
    EXPECT_TRUE(i32_tensor.has_pointee_type());
    auto& i32_pointee = i32_tensor.pointee_type();
    EXPECT_EQ(i32_pointee.type_id(), types::TypeID::Scalar);
    EXPECT_EQ(i32_pointee.primitive_type(), types::PrimitiveType::Int32);
}

/**
 * Test shape conversion to symbolic expressions
 */
TEST(MLIRFrontendTest, ShapeToSymbolic) {
    std::vector<int64_t> shape = {32, 64, 128};
    auto symbolic_shape = MLIRFrontend::shape_to_symbolic(shape);

    EXPECT_EQ(symbolic_shape.size(), 3);
    EXPECT_EQ(symbolic_shape[0]->__str__(), "32");
    EXPECT_EQ(symbolic_shape[1]->__str__(), "64");
    EXPECT_EQ(symbolic_shape[2]->__str__(), "128");
}

/**
 * Test elementwise operation code mapping
 */
TEST(MLIRFrontendTest, ElementwiseOpCodeMapping) {
    // Binary operations
    EXPECT_EQ(MLIRFrontend::get_elementwise_op_code("add"), "Add");
    EXPECT_EQ(MLIRFrontend::get_elementwise_op_code("sub"), "Sub");
    EXPECT_EQ(MLIRFrontend::get_elementwise_op_code("mul"), "Mul");
    EXPECT_EQ(MLIRFrontend::get_elementwise_op_code("div"), "Div");
    EXPECT_EQ(MLIRFrontend::get_elementwise_op_code("pow"), "Pow");
    EXPECT_EQ(MLIRFrontend::get_elementwise_op_code("minimum"), "Minimum");
    EXPECT_EQ(MLIRFrontend::get_elementwise_op_code("maximum"), "Maximum");

    // Unary operations
    EXPECT_EQ(MLIRFrontend::get_elementwise_op_code("abs"), "Abs");
    EXPECT_EQ(MLIRFrontend::get_elementwise_op_code("sqrt"), "Sqrt");
    EXPECT_EQ(MLIRFrontend::get_elementwise_op_code("exp"), "Exp");
    EXPECT_EQ(MLIRFrontend::get_elementwise_op_code("erf"), "Erf");
    EXPECT_EQ(MLIRFrontend::get_elementwise_op_code("sigmoid"), "Sigmoid");
    EXPECT_EQ(MLIRFrontend::get_elementwise_op_code("tanh"), "Tanh");
    EXPECT_EQ(MLIRFrontend::get_elementwise_op_code("relu"), "Relu");
    EXPECT_EQ(MLIRFrontend::get_elementwise_op_code("leaky_relu"), "LeakyRelu");
    EXPECT_EQ(MLIRFrontend::get_elementwise_op_code("elu"), "Elu");

    // Unsupported operation
    EXPECT_THROW(MLIRFrontend::get_elementwise_op_code("unsupported_op"), std::invalid_argument);
}

/**
 * Test reduction operation code mapping
 */
TEST(MLIRFrontendTest, ReduceOpCodeMapping) {
    EXPECT_EQ(MLIRFrontend::get_reduce_op_code("sum"), "Sum");
    EXPECT_EQ(MLIRFrontend::get_reduce_op_code("mean"), "Mean");
    EXPECT_EQ(MLIRFrontend::get_reduce_op_code("std"), "Std");
    EXPECT_EQ(MLIRFrontend::get_reduce_op_code("max"), "Max");
    EXPECT_EQ(MLIRFrontend::get_reduce_op_code("min"), "Min");
    EXPECT_EQ(MLIRFrontend::get_reduce_op_code("softmax"), "Softmax");

    // Unsupported operation
    EXPECT_THROW(MLIRFrontend::get_reduce_op_code("unsupported_reduce"), std::invalid_argument);
}

/**
 * Test operation type classification
 */
TEST(MLIRFrontendTest, OperationClassification) {
    // Unary elementwise operations
    EXPECT_TRUE(MLIRFrontend::is_elementwise_unary("abs"));
    EXPECT_TRUE(MLIRFrontend::is_elementwise_unary("sqrt"));
    EXPECT_TRUE(MLIRFrontend::is_elementwise_unary("relu"));
    EXPECT_FALSE(MLIRFrontend::is_elementwise_unary("add"));
    EXPECT_FALSE(MLIRFrontend::is_elementwise_unary("sum"));

    // Binary elementwise operations
    EXPECT_TRUE(MLIRFrontend::is_elementwise_binary("add"));
    EXPECT_TRUE(MLIRFrontend::is_elementwise_binary("mul"));
    EXPECT_TRUE(MLIRFrontend::is_elementwise_binary("div"));
    EXPECT_FALSE(MLIRFrontend::is_elementwise_binary("abs"));
    EXPECT_FALSE(MLIRFrontend::is_elementwise_binary("sum"));

    // Reduce operations
    EXPECT_TRUE(MLIRFrontend::is_reduce_op("sum"));
    EXPECT_TRUE(MLIRFrontend::is_reduce_op("mean"));
    EXPECT_TRUE(MLIRFrontend::is_reduce_op("max"));
    EXPECT_FALSE(MLIRFrontend::is_reduce_op("add"));
    EXPECT_FALSE(MLIRFrontend::is_reduce_op("abs"));
}
