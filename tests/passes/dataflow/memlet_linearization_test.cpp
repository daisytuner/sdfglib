#include "sdfg/passes/dataflow/memlet_linearization.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/array.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"

using namespace sdfg;

TEST(MemletLinearization, FlattenTwoDimensionalArray) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    auto& block = builder.add_block(builder.subject().root());

    // Create a 3x4 array of int32
    types::Scalar scalar(types::PrimitiveType::Int32);
    types::Array inner_array(scalar, symbolic::integer(4));
    types::Array outer_array(inner_array, symbolic::integer(3));
    types::Pointer pointer_to_array(outer_array);
    types::Pointer opaque_ptr;

    builder.add_container("source", opaque_ptr);
    builder.add_container("ptr", opaque_ptr);

    auto& access_source = builder.add_access(block, "source");
    auto& access_ptr = builder.add_access(block, "ptr");

    // Create a reference memlet with subset [1][2] accessing pointer-to-array
    // This should become [1*4 + 2] = [6] after normalization
    auto& memlet = builder.add_reference_memlet(
        block, access_source, access_ptr, {symbolic::integer(1), symbolic::integer(2)}, pointer_to_array
    );

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::MemletLinearizationPass pass;
    EXPECT_TRUE(pass.run(builder_opt, analysis_manager));
    sdfg = builder_opt.move();

    // Check that base_type is now a pointer to scalar
    auto& new_base_type = memlet.base_type();
    EXPECT_EQ(new_base_type.type_id(), types::TypeID::Pointer);
    auto& new_pointer_type = dynamic_cast<const types::Pointer&>(new_base_type);
    EXPECT_TRUE(new_pointer_type.has_pointee_type());
    EXPECT_EQ(new_pointer_type.pointee_type().type_id(), types::TypeID::Scalar);

    // Check that subset is flattened to [6]
    auto& subset = memlet.subset();
    EXPECT_EQ(subset.size(), 1);
    EXPECT_TRUE(symbolic::eq(subset[0], symbolic::integer(6)));
}

TEST(MemletLinearization, FlattenThreeDimensionalArray) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    auto& block = builder.add_block(builder.subject().root());

    // Create a 2x3x4 array of int32
    types::Scalar scalar(types::PrimitiveType::Int32);
    types::Array array1(scalar, symbolic::integer(4));
    types::Array array2(array1, symbolic::integer(3));
    types::Array array3(array2, symbolic::integer(2));
    types::Pointer pointer_to_array(array3);
    types::Pointer opaque_ptr;

    builder.add_container("source", opaque_ptr);
    builder.add_container("ptr", opaque_ptr);

    auto& access_source = builder.add_access(block, "source");
    auto& access_ptr = builder.add_access(block, "ptr");

    // Create a reference memlet with subset [1][2][3]
    // This should become [1*3*4 + 2*4 + 3] = [12 + 8 + 3] = [23] after normalization
    auto& memlet = builder.add_reference_memlet(
        block,
        access_source,
        access_ptr,
        {symbolic::integer(1), symbolic::integer(2), symbolic::integer(3)},
        pointer_to_array
    );

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::MemletLinearizationPass pass;
    EXPECT_TRUE(pass.run(builder_opt, analysis_manager));
    sdfg = builder_opt.move();

    // Check that base_type is now a pointer to scalar
    auto& new_base_type = memlet.base_type();
    EXPECT_EQ(new_base_type.type_id(), types::TypeID::Pointer);
    auto& new_pointer_type = dynamic_cast<const types::Pointer&>(new_base_type);
    EXPECT_TRUE(new_pointer_type.has_pointee_type());
    EXPECT_EQ(new_pointer_type.pointee_type().type_id(), types::TypeID::Scalar);

    // Check that subset is flattened to [23]
    auto& subset = memlet.subset();
    EXPECT_EQ(subset.size(), 1);
    EXPECT_TRUE(symbolic::eq(subset[0], symbolic::integer(23)));
}

TEST(MemletLinearization, FlattenWithSymbolicIndex) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    auto& block = builder.add_block(builder.subject().root());

    // Create a 3x4 array of int32
    types::Scalar scalar(types::PrimitiveType::Int32);
    types::Array inner_array(scalar, symbolic::integer(4));
    types::Array outer_array(inner_array, symbolic::integer(3));
    types::Pointer pointer_to_array(outer_array);
    types::Pointer opaque_ptr;

    builder.add_container("source", opaque_ptr);
    builder.add_container("ptr", opaque_ptr);

    auto& access_source = builder.add_access(block, "source");
    auto& access_ptr = builder.add_access(block, "ptr");

    // Create a reference memlet with subset [i][j]
    // This should become [i*4 + j] after normalization
    symbolic::Expression i_sym = symbolic::symbol("i");
    symbolic::Expression j_sym = symbolic::symbol("j");
    auto& memlet = builder.add_reference_memlet(block, access_source, access_ptr, {i_sym, j_sym}, pointer_to_array);

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::MemletLinearizationPass pass;
    EXPECT_TRUE(pass.run(builder_opt, analysis_manager));
    sdfg = builder_opt.move();

    // Check that base_type is now a pointer to scalar
    auto& new_base_type = memlet.base_type();
    EXPECT_EQ(new_base_type.type_id(), types::TypeID::Pointer);
    auto& new_pointer_type = dynamic_cast<const types::Pointer&>(new_base_type);
    EXPECT_TRUE(new_pointer_type.has_pointee_type());
    EXPECT_EQ(new_pointer_type.pointee_type().type_id(), types::TypeID::Scalar);

    // Check that subset is flattened to [i*4 + j]
    auto& subset = memlet.subset();
    EXPECT_EQ(subset.size(), 1);

    // Verify the expression: i*4 + j
    symbolic::Expression expected = symbolic::add(symbolic::mul(i_sym, symbolic::integer(4)), j_sym);
    EXPECT_TRUE(symbolic::eq(subset[0], expected));
}

TEST(MemletLinearization, NoChangeForScalarPointer) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    auto& block = builder.add_block(builder.subject().root());

    // Create a pointer to scalar (not nested arrays)
    types::Scalar scalar(types::PrimitiveType::Int32);
    types::Pointer pointer_to_scalar(scalar);
    types::Pointer opaque_ptr;

    builder.add_container("source", opaque_ptr);
    builder.add_container("ptr", opaque_ptr);

    auto& access_source = builder.add_access(block, "source");
    auto& access_ptr = builder.add_access(block, "ptr");

    auto& memlet =
        builder.add_reference_memlet(block, access_source, access_ptr, {symbolic::integer(0)}, pointer_to_scalar);

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::MemletLinearizationPass pass;
    EXPECT_FALSE(pass.run(builder_opt, analysis_manager)); // Should return false (no changes)
    sdfg = builder_opt.move();

    // Check that base_type is still a pointer to scalar
    auto& base_type = memlet.base_type();
    EXPECT_EQ(base_type.type_id(), types::TypeID::Pointer);
    auto& pointer_type = dynamic_cast<const types::Pointer&>(base_type);
    EXPECT_TRUE(pointer_type.has_pointee_type());
    EXPECT_EQ(pointer_type.pointee_type().type_id(), types::TypeID::Scalar);

    // Check that subset is unchanged
    auto& subset = memlet.subset();
    EXPECT_EQ(subset.size(), 1);
    EXPECT_TRUE(symbolic::eq(subset[0], symbolic::integer(0)));
}

TEST(MemletLinearization, PartialIndex) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    auto& block = builder.add_block(builder.subject().root());

    // Create a 3x4x5 array of int32
    types::Scalar scalar(types::PrimitiveType::Int32);
    types::Array array1(scalar, symbolic::integer(5));
    types::Array array2(array1, symbolic::integer(4));
    types::Array array3(array2, symbolic::integer(3));
    types::Pointer pointer_to_array(array3);
    types::Pointer opaque_ptr;

    builder.add_container("source", opaque_ptr);
    builder.add_container("ptr", opaque_ptr);

    auto& access_source = builder.add_access(block, "source");
    auto& access_ptr = builder.add_access(block, "ptr");

    // Access only first dimension [1] - should become [1*4*5] = [20]
    auto& memlet =
        builder.add_reference_memlet(block, access_source, access_ptr, {symbolic::integer(1)}, pointer_to_array);

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::MemletLinearizationPass pass;
    EXPECT_TRUE(pass.run(builder_opt, analysis_manager));
    sdfg = builder_opt.move();

    // Check that base_type is now a pointer to scalar
    auto& new_base_type = memlet.base_type();
    EXPECT_EQ(new_base_type.type_id(), types::TypeID::Pointer);
    auto& new_pointer_type = dynamic_cast<const types::Pointer&>(new_base_type);
    EXPECT_TRUE(new_pointer_type.has_pointee_type());
    EXPECT_EQ(new_pointer_type.pointee_type().type_id(), types::TypeID::Scalar);

    // Check that subset is flattened to [20]
    auto& subset = memlet.subset();
    EXPECT_EQ(subset.size(), 1);
    EXPECT_TRUE(symbolic::eq(subset[0], symbolic::integer(20)));
}

TEST(MemletLinearization, FlattenSingleDimensionArray) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    auto& block = builder.add_block(builder.subject().root());

    // Create a 10 element array of int32
    types::Scalar scalar(types::PrimitiveType::Int32);
    types::Array array(scalar, symbolic::integer(10));
    types::Pointer pointer_to_array(array);
    types::Pointer opaque_ptr;

    builder.add_container("source", opaque_ptr);
    builder.add_container("ptr", opaque_ptr);

    auto& access_source = builder.add_access(block, "source");
    auto& access_ptr = builder.add_access(block, "ptr");

    // Access [5]
    auto& memlet =
        builder.add_reference_memlet(block, access_source, access_ptr, {symbolic::integer(5)}, pointer_to_array);

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::MemletLinearizationPass pass;
    EXPECT_TRUE(pass.run(builder_opt, analysis_manager));
    sdfg = builder_opt.move();

    // Check that base_type is now a pointer to scalar
    auto& new_base_type = memlet.base_type();
    EXPECT_EQ(new_base_type.type_id(), types::TypeID::Pointer);
    auto& new_pointer_type = dynamic_cast<const types::Pointer&>(new_base_type);
    EXPECT_TRUE(new_pointer_type.has_pointee_type());
    EXPECT_EQ(new_pointer_type.pointee_type().type_id(), types::TypeID::Scalar);

    // Check that subset is [5] (unchanged in value but still normalized)
    auto& subset = memlet.subset();
    EXPECT_EQ(subset.size(), 1);
    EXPECT_TRUE(symbolic::eq(subset[0], symbolic::integer(5)));
}
