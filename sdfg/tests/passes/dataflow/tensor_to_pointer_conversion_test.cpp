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
