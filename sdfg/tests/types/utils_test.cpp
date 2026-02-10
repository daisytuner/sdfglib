#include "sdfg/types/utils.h"

#include <gtest/gtest.h>

#include "sdfg/builder/sdfg_builder.h"
#include "sdfg/types/array.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/type.h"

using namespace sdfg;

TEST(TypeInferenceTest, Identity) {
    builder::SDFGBuilder builder("test", FunctionType_CPU);
    auto& function = builder.subject();

    data_flow::Subset subset = {};

    types::Scalar scalar_type(types::PrimitiveType::Int32);
    auto inferred1 = types::infer_type(function, scalar_type, subset);
    EXPECT_EQ(*inferred1, scalar_type);

    types::Array array_type(scalar_type, symbolic::integer(10));
    auto inferred2 = types::infer_type(function, array_type, subset);
    EXPECT_EQ(*inferred2, array_type);

    types::Pointer pointer_type(scalar_type);
    auto inferred3 = types::infer_type(function, pointer_type, subset);
    EXPECT_EQ(*inferred3, pointer_type);

    types::Structure structure_type("test");
    auto inferred4 = types::infer_type(function, structure_type, subset);
    EXPECT_EQ(*inferred4, structure_type);
}

TEST(TypeInferenceTest, Scalar) {
    builder::SDFGBuilder builder("test", FunctionType_CPU);
    auto& function = builder.subject();

    types::Scalar scalar_type(types::PrimitiveType::Int32);
    data_flow::Subset subset = {};
    auto inferred = types::infer_type(function, scalar_type, subset);
    EXPECT_EQ(*inferred, scalar_type);
}

TEST(TypeInferenceTest, ElementType) {
    builder::SDFGBuilder builder("test", FunctionType_CPU);
    auto& function = builder.subject();

    types::Scalar scalar_type(types::PrimitiveType::Int32);
    types::Array array_type(scalar_type, symbolic::integer(10));

    data_flow::Subset subset = {symbolic::integer(0)};
    auto inferred = types::infer_type(function, array_type, subset);
    EXPECT_EQ(*inferred, scalar_type);
}

TEST(TypeInferenceTest, PointeeType) {
    builder::SDFGBuilder builder("test", FunctionType_CPU);
    auto& function = builder.subject();

    types::Scalar scalar_type(types::PrimitiveType::Int32);
    types::Pointer pointer_type(scalar_type);

    data_flow::Subset subset = {symbolic::integer(0)};
    auto inferred = types::infer_type(function, pointer_type, subset);
    EXPECT_EQ(*inferred, scalar_type);
}

TEST(TypeInferenceTest, StructureMember) {
    builder::SDFGBuilder builder("test", FunctionType_CPU);

    auto& sd = builder.add_structure("test", false);

    types::Scalar scalar_type1(types::PrimitiveType::Int32);
    sd.add_member(scalar_type1);

    types::Scalar scalar_type2(types::PrimitiveType::Float);
    sd.add_member(scalar_type2);

    auto& function = builder.subject();

    types::Structure structure_type("test");
    data_flow::Subset subset = {symbolic::integer(1)};
    auto inferred = types::infer_type(function, structure_type, subset);
    EXPECT_EQ(*inferred, scalar_type2);
}

TEST(TypeInferenceTest, Tensor) {
    builder::SDFGBuilder builder("test", FunctionType_CPU);
    auto& function = builder.subject();

    types::Scalar scalar_type(types::PrimitiveType::Int32);
    symbolic::MultiExpression shape = {symbolic::integer(10)};
    types::Tensor tensor_1d(scalar_type, shape);

    symbolic::MultiExpression shape_2d = {symbolic::integer(10), symbolic::integer(20)};
    types::Tensor tensor_2d(scalar_type, shape_2d);

    data_flow::Subset subset = {symbolic::integer(0)};
    auto inferred = types::infer_type(function, tensor_1d, subset);
    EXPECT_EQ(*inferred, scalar_type);

    auto inferred2 = types::infer_type(function, tensor_2d, subset);
    EXPECT_TRUE(inferred2->type_id() == types::TypeID::Tensor);

    auto& inferred_tensor = static_cast<types::Tensor&>(*inferred2);
    EXPECT_EQ(inferred_tensor.element_type(), scalar_type);
    EXPECT_EQ(inferred_tensor.shape().size(), 1);
    EXPECT_TRUE(symbolic::eq(inferred_tensor.shape().at(0), symbolic::integer(20)));
    EXPECT_EQ(inferred_tensor.strides().size(), 1);
    EXPECT_TRUE(symbolic::eq(inferred_tensor.strides().at(0), symbolic::integer(1)));
}

TEST(PeelToNextElement, Scalar) {
    types::Scalar scalar_type(types::PrimitiveType::Int32);
    auto* peeled = types::peel_to_next_element(scalar_type);
    EXPECT_EQ(peeled, &scalar_type);
}

TEST(PeelToNextElement, Array) {
    types::Scalar scalar_type(types::PrimitiveType::Int32);
    types::Array array_type(scalar_type, symbolic::integer(10));
    auto* peeled = types::peel_to_next_element(array_type);
    EXPECT_EQ(*peeled, scalar_type);
}

TEST(PeelToNextElement, Pointer) {
    types::Scalar scalar_type(types::PrimitiveType::Int32);
    types::Pointer pointer_type(scalar_type);
    auto* peeled = types::peel_to_next_element(pointer_type);
    EXPECT_EQ(*peeled, scalar_type);
}

TEST(PeelToNextElement, Tensor) {
    types::Scalar scalar_type(types::PrimitiveType::Int32);
    symbolic::MultiExpression shape = {symbolic::integer(10)};
    types::Tensor tensor_type(scalar_type, shape);
    auto* peeled = types::peel_to_next_element(tensor_type);
    EXPECT_EQ(*peeled, scalar_type);
}

TEST(PeelToNextElement, NestedArray) {
    types::Scalar scalar_type(types::PrimitiveType::Int32);
    types::Array array_type1(scalar_type, symbolic::integer(10));
    types::Array array_type2(array_type1, symbolic::integer(20));
    auto* peeled = types::peel_to_next_element(array_type2);
    EXPECT_EQ(*peeled, array_type1);
}

TEST(PeelToNextElement, NestedPointer) {
    types::Scalar scalar_type(types::PrimitiveType::Int32);
    types::Pointer pointer_type1(scalar_type);
    types::Pointer pointer_type2(types::StorageType::CPU_Heap(), 8, "", pointer_type1);
    auto* peeled = types::peel_to_next_element(pointer_type2);
    EXPECT_EQ(*peeled, pointer_type1);
}
