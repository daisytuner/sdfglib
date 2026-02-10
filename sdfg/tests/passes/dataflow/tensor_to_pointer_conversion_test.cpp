#include "sdfg/passes/dataflow/tensor_to_pointer_conversion.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/tensor.h"

using namespace sdfg;

TEST(TensorToPointerConversion, Tensor1D) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    auto& block = builder.add_block(builder.subject().root());

    types::Scalar scalar(types::PrimitiveType::Int32);

    symbolic::MultiExpression shape = {symbolic::integer(4)};
    symbolic::MultiExpression strides = {symbolic::one()};
    types::Tensor tensor_1d(scalar, shape, strides, symbolic::zero());

    types::Pointer opaque_ptr;
    builder.add_container("A", opaque_ptr);
    builder.add_container("B", opaque_ptr);

    auto& A_in = builder.add_access(block, "A");
    auto& B_out = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    auto& iedge = builder.add_computational_memlet(block, A_in, tasklet, "_in", {symbolic::one()}, tensor_1d);
    auto& oedge = builder.add_computational_memlet(block, tasklet, "_out", B_out, {symbolic::zero()}, tensor_1d);

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::TensorToPointerConversionPass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    EXPECT_EQ(iedge.base_type().type_id(), types::TypeID::Pointer);
    auto& pointer_type = dynamic_cast<const types::Pointer&>(iedge.base_type());
    EXPECT_TRUE(pointer_type.has_pointee_type());
    EXPECT_EQ(pointer_type.pointee_type().type_id(), types::TypeID::Scalar);
    EXPECT_EQ(pointer_type.primitive_type(), types::PrimitiveType::Int32);
    EXPECT_EQ(iedge.subset().size(), 1);
    EXPECT_TRUE(symbolic::eq(iedge.subset()[0], symbolic::one()));

    EXPECT_EQ(oedge.base_type().type_id(), types::TypeID::Pointer);
    auto& pointer_type_out = dynamic_cast<const types::Pointer&>(oedge.base_type());
    EXPECT_TRUE(pointer_type_out.has_pointee_type());
    EXPECT_EQ(pointer_type_out.pointee_type().type_id(), types::TypeID::Scalar);
    EXPECT_EQ(pointer_type_out.primitive_type(), types::PrimitiveType::Int32);
    EXPECT_EQ(oedge.subset().size(), 1);
    EXPECT_TRUE(symbolic::eq(oedge.subset()[0], symbolic::zero()));
}

TEST(TensorToPointerConversion, Tensor1D_Flip) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    auto& block = builder.add_block(builder.subject().root());

    types::Scalar scalar(types::PrimitiveType::Int32);

    symbolic::MultiExpression shape = {symbolic::integer(4)};
    symbolic::MultiExpression strides = {symbolic::one()};
    types::Tensor tensor_1d(scalar, shape, strides, symbolic::zero());
    auto tensor_1d_flipped = tensor_1d.flip(0);

    types::Pointer opaque_ptr;
    builder.add_container("A", opaque_ptr);
    builder.add_container("B", opaque_ptr);

    auto& A_in = builder.add_access(block, "A");
    auto& B_out = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    auto& iedge = builder.add_computational_memlet(block, A_in, tasklet, "_in", {symbolic::one()}, *tensor_1d_flipped);
    auto& oedge =
        builder.add_computational_memlet(block, tasklet, "_out", B_out, {symbolic::zero()}, *tensor_1d_flipped);

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::TensorToPointerConversionPass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    EXPECT_EQ(iedge.base_type().type_id(), types::TypeID::Pointer);
    auto& pointer_type = dynamic_cast<const types::Pointer&>(iedge.base_type());
    EXPECT_TRUE(pointer_type.has_pointee_type());
    EXPECT_EQ(pointer_type.pointee_type().type_id(), types::TypeID::Scalar);
    EXPECT_EQ(pointer_type.primitive_type(), types::PrimitiveType::Int32);
    EXPECT_EQ(iedge.subset().size(), 1);
    EXPECT_TRUE(symbolic::eq(iedge.subset()[0], symbolic::integer(2)));

    EXPECT_EQ(oedge.base_type().type_id(), types::TypeID::Pointer);
    auto& pointer_type_out = dynamic_cast<const types::Pointer&>(oedge.base_type());
    EXPECT_TRUE(pointer_type_out.has_pointee_type());
    EXPECT_EQ(pointer_type_out.pointee_type().type_id(), types::TypeID::Scalar);
    EXPECT_EQ(pointer_type_out.primitive_type(), types::PrimitiveType::Int32);
    EXPECT_EQ(oedge.subset().size(), 1);
    EXPECT_TRUE(symbolic::eq(oedge.subset()[0], symbolic::integer(3)));
}

TEST(TensorToPointerConversion, Tensor2D_NonCLayout) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    auto& block = builder.add_block(builder.subject().root());

    types::Scalar scalar(types::PrimitiveType::Int32);

    // 2D tensor with shape [4, 8] and column-major (Fortran) strides [1, 4]
    // In C-layout, strides would be [8, 1]
    symbolic::MultiExpression shape = {symbolic::integer(4), symbolic::integer(8)};
    symbolic::MultiExpression strides = {symbolic::one(), symbolic::integer(4)};
    types::Tensor tensor_2d(scalar, shape, strides, symbolic::zero());

    types::Pointer opaque_ptr;
    builder.add_container("A", opaque_ptr);
    builder.add_container("B", opaque_ptr);

    auto& A_in = builder.add_access(block, "A");
    auto& B_out = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    // Access at index [2, 3] -> linear offset = 2*1 + 3*4 = 14
    auto& iedge = builder.add_computational_memlet(
        block, A_in, tasklet, "_in", {symbolic::integer(2), symbolic::integer(3)}, tensor_2d
    );
    // Access at index [1, 5] -> linear offset = 1*1 + 5*4 = 21
    auto& oedge =
        builder
            .add_computational_memlet(block, tasklet, "_out", B_out, {symbolic::one(), symbolic::integer(5)}, tensor_2d);

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::TensorToPointerConversionPass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    EXPECT_EQ(iedge.base_type().type_id(), types::TypeID::Pointer);
    auto& pointer_type = dynamic_cast<const types::Pointer&>(iedge.base_type());
    EXPECT_TRUE(pointer_type.has_pointee_type());
    EXPECT_EQ(pointer_type.pointee_type().type_id(), types::TypeID::Scalar);
    EXPECT_EQ(pointer_type.primitive_type(), types::PrimitiveType::Int32);
    EXPECT_EQ(iedge.subset().size(), 1);
    // Linear offset for [2, 3] with strides [1, 4] = 2*1 + 3*4 = 14
    EXPECT_TRUE(symbolic::eq(iedge.subset()[0], symbolic::integer(14)));

    EXPECT_EQ(oedge.base_type().type_id(), types::TypeID::Pointer);
    auto& pointer_type_out = dynamic_cast<const types::Pointer&>(oedge.base_type());
    EXPECT_TRUE(pointer_type_out.has_pointee_type());
    EXPECT_EQ(pointer_type_out.pointee_type().type_id(), types::TypeID::Scalar);
    EXPECT_EQ(pointer_type_out.primitive_type(), types::PrimitiveType::Int32);
    EXPECT_EQ(oedge.subset().size(), 1);
    // Linear offset for [1, 5] with strides [1, 4] = 1*1 + 5*4 = 21
    EXPECT_TRUE(symbolic::eq(oedge.subset()[0], symbolic::integer(21)));
}

TEST(TensorToPointerConversion, Tensor3D_NonCLayout) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    auto& block = builder.add_block(builder.subject().root());

    types::Scalar scalar(types::PrimitiveType::Int32);

    // 3D tensor with shape [2, 3, 4] and non-C-layout strides [12, 1, 3]
    // In C-layout, strides would be [12, 4, 1]
    // This represents a layout where dimension 1 is contiguous
    symbolic::MultiExpression shape = {symbolic::integer(2), symbolic::integer(3), symbolic::integer(4)};
    symbolic::MultiExpression strides = {symbolic::integer(12), symbolic::one(), symbolic::integer(3)};
    types::Tensor tensor_3d(scalar, shape, strides, symbolic::zero());

    types::Pointer opaque_ptr;
    builder.add_container("A", opaque_ptr);
    builder.add_container("B", opaque_ptr);

    auto& A_in = builder.add_access(block, "A");
    auto& B_out = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    // Access at index [1, 2, 3] -> linear offset = 1*12 + 2*1 + 3*3 = 12 + 2 + 9 = 23
    auto& iedge = builder.add_computational_memlet(
        block, A_in, tasklet, "_in", {symbolic::one(), symbolic::integer(2), symbolic::integer(3)}, tensor_3d
    );
    // Access at index [0, 1, 2] -> linear offset = 0*12 + 1*1 + 2*3 = 0 + 1 + 6 = 7
    auto& oedge = builder.add_computational_memlet(
        block, tasklet, "_out", B_out, {symbolic::zero(), symbolic::one(), symbolic::integer(2)}, tensor_3d
    );

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::TensorToPointerConversionPass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    EXPECT_EQ(iedge.base_type().type_id(), types::TypeID::Pointer);
    auto& pointer_type = dynamic_cast<const types::Pointer&>(iedge.base_type());
    EXPECT_TRUE(pointer_type.has_pointee_type());
    EXPECT_EQ(pointer_type.pointee_type().type_id(), types::TypeID::Scalar);
    EXPECT_EQ(pointer_type.primitive_type(), types::PrimitiveType::Int32);
    EXPECT_EQ(iedge.subset().size(), 1);
    // Linear offset for [1, 2, 3] with strides [12, 1, 3] = 1*12 + 2*1 + 3*3 = 23
    EXPECT_TRUE(symbolic::eq(iedge.subset()[0], symbolic::integer(23)));

    EXPECT_EQ(oedge.base_type().type_id(), types::TypeID::Pointer);
    auto& pointer_type_out = dynamic_cast<const types::Pointer&>(oedge.base_type());
    EXPECT_TRUE(pointer_type_out.has_pointee_type());
    EXPECT_EQ(pointer_type_out.pointee_type().type_id(), types::TypeID::Scalar);
    EXPECT_EQ(pointer_type_out.primitive_type(), types::PrimitiveType::Int32);
    EXPECT_EQ(oedge.subset().size(), 1);
    // Linear offset for [0, 1, 2] with strides [12, 1, 3] = 0*12 + 1*1 + 2*3 = 7
    EXPECT_TRUE(symbolic::eq(oedge.subset()[0], symbolic::integer(7)));
}
