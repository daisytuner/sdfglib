#include "sdfg/passes/dataflow/memlet_base_type_normalization.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/array.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"

using namespace sdfg;

TEST(MemletBaseTypeNormalization, FlattenTwoDimensionalArray) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    auto& block = builder.add_block(builder.subject().root());

    // Create a 3x4 array of int32
    types::Scalar scalar(types::PrimitiveType::Int32);
    types::Array inner_array(scalar, symbolic::integer(4));
    types::Array outer_array(inner_array, symbolic::integer(3));
    types::Pointer pointer_to_array(outer_array);
    types::Scalar result_scalar(types::PrimitiveType::Int32);

    builder.add_container("ptr", pointer_to_array);
    builder.add_container("result", result_scalar);

    auto& access_ptr = builder.add_access(block, "ptr");
    auto& access_result = builder.add_access(block, "result");

    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});

    // Access [1][2] - should become [1*4 + 2] = [6]
    auto& memlet_in =
        builder
            .add_computational_memlet(block, access_ptr, tasklet, "_in", {symbolic::integer(1), symbolic::integer(2)});

    auto& memlet_out = builder.add_computational_memlet(block, tasklet, "_out", access_result, {});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::MemletBaseTypeNormalization pass;
    EXPECT_TRUE(pass.run_pass(builder_opt, analysis_manager));
    sdfg = builder_opt.move();

    // Check that base_type is now a pointer to scalar
    auto& new_base_type = memlet_in.base_type();
    EXPECT_EQ(new_base_type.type_id(), types::TypeID::Pointer);
    auto& new_pointer_type = dynamic_cast<const types::Pointer&>(new_base_type);
    EXPECT_TRUE(new_pointer_type.has_pointee_type());
    EXPECT_EQ(new_pointer_type.pointee_type().type_id(), types::TypeID::Scalar);

    // Check that subset is flattened to [6]
    auto& subset = memlet_in.subset();
    EXPECT_EQ(subset.size(), 1);
    EXPECT_TRUE(symbolic::eq(subset[0], symbolic::integer(6)));
}

TEST(MemletBaseTypeNormalization, FlattenThreeDimensionalArray) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    auto& block = builder.add_block(builder.subject().root());

    // Create a 2x3x4 array of int32
    types::Scalar scalar(types::PrimitiveType::Int32);
    types::Array array1(scalar, symbolic::integer(4));
    types::Array array2(array1, symbolic::integer(3));
    types::Array array3(array2, symbolic::integer(2));
    types::Pointer pointer_to_array(array3);
    types::Scalar result_scalar(types::PrimitiveType::Int32);

    builder.add_container("ptr", pointer_to_array);
    builder.add_container("result", result_scalar);

    auto& access_ptr = builder.add_access(block, "ptr");
    auto& access_result = builder.add_access(block, "result");

    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});

    // Access [1][2][3] - should become [1*3*4 + 2*4 + 3] = [12 + 8 + 3] = [23]
    auto& memlet_in = builder.add_computational_memlet(
        block, access_ptr, tasklet, "_in", {symbolic::integer(1), symbolic::integer(2), symbolic::integer(3)}
    );

    auto& memlet_out = builder.add_computational_memlet(block, tasklet, "_out", access_result, {});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::MemletBaseTypeNormalization pass;
    EXPECT_TRUE(pass.run_pass(builder_opt, analysis_manager));
    sdfg = builder_opt.move();

    // Check that base_type is now a pointer to scalar
    auto& new_base_type = memlet_in.base_type();
    EXPECT_EQ(new_base_type.type_id(), types::TypeID::Pointer);
    auto& new_pointer_type = dynamic_cast<const types::Pointer&>(new_base_type);
    EXPECT_TRUE(new_pointer_type.has_pointee_type());
    EXPECT_EQ(new_pointer_type.pointee_type().type_id(), types::TypeID::Scalar);

    // Check that subset is flattened to [23]
    auto& subset = memlet_in.subset();
    EXPECT_EQ(subset.size(), 1);
    EXPECT_TRUE(symbolic::eq(subset[0], symbolic::integer(23)));
}

TEST(MemletBaseTypeNormalization, FlattenWithSymbolicIndex) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    auto& block = builder.add_block(builder.subject().root());

    // Create a 3x4 array of int32
    types::Scalar scalar(types::PrimitiveType::Int32);
    types::Array inner_array(scalar, symbolic::integer(4));
    types::Array outer_array(inner_array, symbolic::integer(3));
    types::Pointer pointer_to_array(outer_array);
    types::Scalar result_scalar(types::PrimitiveType::Int32);

    builder.add_container("ptr", pointer_to_array);
    builder.add_container("result", result_scalar);

    auto& access_ptr = builder.add_access(block, "ptr");
    auto& access_result = builder.add_access(block, "result");

    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});

    // Access [i][j] - should become [i*4 + j]
    symbolic::Expression i_sym = symbolic::symbol("i");
    symbolic::Expression j_sym = symbolic::symbol("j");
    auto& memlet_in = builder.add_computational_memlet(block, access_ptr, tasklet, "_in", {i_sym, j_sym});

    auto& memlet_out = builder.add_computational_memlet(block, tasklet, "_out", access_result, {});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::MemletBaseTypeNormalization pass;
    EXPECT_TRUE(pass.run_pass(builder_opt, analysis_manager));
    sdfg = builder_opt.move();

    // Check that base_type is now a pointer to scalar
    auto& new_base_type = memlet_in.base_type();
    EXPECT_EQ(new_base_type.type_id(), types::TypeID::Pointer);
    auto& new_pointer_type = dynamic_cast<const types::Pointer&>(new_base_type);
    EXPECT_TRUE(new_pointer_type.has_pointee_type());
    EXPECT_EQ(new_pointer_type.pointee_type().type_id(), types::TypeID::Scalar);

    // Check that subset is flattened to [i*4 + j]
    auto& subset = memlet_in.subset();
    EXPECT_EQ(subset.size(), 1);

    // Verify the expression: i*4 + j
    symbolic::Expression expected = symbolic::add(symbolic::mul(i_sym, symbolic::integer(4)), j_sym);
    EXPECT_TRUE(symbolic::eq(subset[0], expected));
}

TEST(MemletBaseTypeNormalization, NoChangeForScalarPointer) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    auto& block = builder.add_block(builder.subject().root());

    // Create a pointer to scalar (not nested arrays)
    types::Scalar scalar(types::PrimitiveType::Int32);
    types::Pointer pointer_to_scalar(scalar);
    types::Scalar result_scalar(types::PrimitiveType::Int32);

    builder.add_container("ptr", pointer_to_scalar);
    builder.add_container("result", result_scalar);

    auto& access_ptr = builder.add_access(block, "ptr");
    auto& access_result = builder.add_access(block, "result");

    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});

    auto& memlet_in = builder.add_computational_memlet(block, access_ptr, tasklet, "_in", {symbolic::integer(0)});

    auto& memlet_out = builder.add_computational_memlet(block, tasklet, "_out", access_result, {});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::MemletBaseTypeNormalization pass;
    EXPECT_FALSE(pass.run_pass(builder_opt, analysis_manager)); // Should return false (no changes)
    sdfg = builder_opt.move();

    // Check that base_type is still a pointer to scalar
    auto& base_type = memlet_in.base_type();
    EXPECT_EQ(base_type.type_id(), types::TypeID::Pointer);
    auto& pointer_type = dynamic_cast<const types::Pointer&>(base_type);
    EXPECT_TRUE(pointer_type.has_pointee_type());
    EXPECT_EQ(pointer_type.pointee_type().type_id(), types::TypeID::Scalar);

    // Check that subset is unchanged
    auto& subset = memlet_in.subset();
    EXPECT_EQ(subset.size(), 1);
    EXPECT_TRUE(symbolic::eq(subset[0], symbolic::integer(0)));
}

TEST(MemletBaseTypeNormalization, PartialIndex) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    auto& block = builder.add_block(builder.subject().root());

    // Create a 3x4x5 array of int32
    types::Scalar scalar(types::PrimitiveType::Int32);
    types::Array array1(scalar, symbolic::integer(5));
    types::Array array2(array1, symbolic::integer(4));
    types::Array array3(array2, symbolic::integer(3));
    types::Pointer pointer_to_array(array3);
    types::Scalar result_scalar(types::PrimitiveType::Int32);

    builder.add_container("ptr", pointer_to_array);
    builder.add_container("result", result_scalar);

    auto& access_ptr = builder.add_access(block, "ptr");
    auto& access_result = builder.add_access(block, "result");

    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});

    // Access only first dimension [1] - should become [1*4*5] = [20]
    auto& memlet_in = builder.add_computational_memlet(block, access_ptr, tasklet, "_in", {symbolic::integer(1)});

    auto& memlet_out = builder.add_computational_memlet(block, tasklet, "_out", access_result, {});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::MemletBaseTypeNormalization pass;
    EXPECT_TRUE(pass.run_pass(builder_opt, analysis_manager));
    sdfg = builder_opt.move();

    // Check that base_type is now a pointer to scalar
    auto& new_base_type = memlet_in.base_type();
    EXPECT_EQ(new_base_type.type_id(), types::TypeID::Pointer);
    auto& new_pointer_type = dynamic_cast<const types::Pointer&>(new_base_type);
    EXPECT_TRUE(new_pointer_type.has_pointee_type());
    EXPECT_EQ(new_pointer_type.pointee_type().type_id(), types::TypeID::Scalar);

    // Check that subset is flattened to [20]
    auto& subset = memlet_in.subset();
    EXPECT_EQ(subset.size(), 1);
    EXPECT_TRUE(symbolic::eq(subset[0], symbolic::integer(20)));
}

TEST(MemletBaseTypeNormalization, FlattenSingleDimensionArray) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    auto& block = builder.add_block(builder.subject().root());

    // Create a 10 element array of int32
    types::Scalar scalar(types::PrimitiveType::Int32);
    types::Array array(scalar, symbolic::integer(10));
    types::Pointer pointer_to_array(array);
    types::Scalar result_scalar(types::PrimitiveType::Int32);

    builder.add_container("ptr", pointer_to_array);
    builder.add_container("result", result_scalar);

    auto& access_ptr = builder.add_access(block, "ptr");
    auto& access_result = builder.add_access(block, "result");

    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});

    // Access [5]
    auto& memlet_in = builder.add_computational_memlet(block, access_ptr, tasklet, "_in", {symbolic::integer(5)});

    auto& memlet_out = builder.add_computational_memlet(block, tasklet, "_out", access_result, {});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::MemletBaseTypeNormalization pass;
    EXPECT_TRUE(pass.run_pass(builder_opt, analysis_manager));
    sdfg = builder_opt.move();

    // Check that base_type is now a pointer to scalar
    auto& new_base_type = memlet_in.base_type();
    EXPECT_EQ(new_base_type.type_id(), types::TypeID::Pointer);
    auto& new_pointer_type = dynamic_cast<const types::Pointer&>(new_base_type);
    EXPECT_TRUE(new_pointer_type.has_pointee_type());
    EXPECT_EQ(new_pointer_type.pointee_type().type_id(), types::TypeID::Scalar);

    // Check that subset is [5] (unchanged in value but still normalized)
    auto& subset = memlet_in.subset();
    EXPECT_EQ(subset.size(), 1);
    EXPECT_TRUE(symbolic::eq(subset[0], symbolic::integer(5)));
}
