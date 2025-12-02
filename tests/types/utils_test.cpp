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
    auto& inferred1 = types::infer_type(function, scalar_type, subset);
    EXPECT_EQ(inferred1, scalar_type);

    types::Array array_type(scalar_type, symbolic::integer(10));
    auto& inferred2 = types::infer_type(function, array_type, subset);
    EXPECT_EQ(inferred2, array_type);

    types::Pointer pointer_type(scalar_type);
    auto& inferred3 = types::infer_type(function, pointer_type, subset);
    EXPECT_EQ(inferred3, pointer_type);

    types::Structure structure_type("test");
    auto& inferred4 = types::infer_type(function, structure_type, subset);
    EXPECT_EQ(inferred4, structure_type);
}

TEST(TypeInferenceTest, Scalar) {
    builder::SDFGBuilder builder("test", FunctionType_CPU);
    auto& function = builder.subject();

    types::Scalar scalar_type(types::PrimitiveType::Int32);
    data_flow::Subset subset = {};
    auto& inferred = types::infer_type(function, scalar_type, subset);
    EXPECT_EQ(inferred, scalar_type);
}

TEST(TypeInferenceTest, ElementType) {
    builder::SDFGBuilder builder("test", FunctionType_CPU);
    auto& function = builder.subject();

    types::Scalar scalar_type(types::PrimitiveType::Int32);
    types::Array array_type(scalar_type, symbolic::integer(10));

    data_flow::Subset subset = {symbolic::integer(0)};
    auto& inferred = types::infer_type(function, array_type, subset);
    EXPECT_EQ(inferred, scalar_type);
}

TEST(TypeInferenceTest, PointeeType) {
    builder::SDFGBuilder builder("test", FunctionType_CPU);
    auto& function = builder.subject();

    types::Scalar scalar_type(types::PrimitiveType::Int32);
    types::Pointer pointer_type(scalar_type);

    data_flow::Subset subset = {symbolic::integer(0)};
    auto& inferred = types::infer_type(function, pointer_type, subset);
    EXPECT_EQ(inferred, scalar_type);
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
    auto& inferred = types::infer_type(function, structure_type, subset);
    EXPECT_EQ(inferred, scalar_type2);
}

TEST(TypeComparisonTest, Scalar) {
    types::Scalar float_type(types::PrimitiveType::Float);
    types::Scalar int_type(types::PrimitiveType::Int32);

    EXPECT_EQ(types::compare_types(float_type, int_type), types::TypeCompare::INCOMPATIBLE);
    EXPECT_EQ(types::compare_types(float_type, float_type), types::TypeCompare::EQUAL);

    types::Scalar float_type2(types::PrimitiveType::Float);
    EXPECT_EQ(types::compare_types(float_type, float_type2), types::TypeCompare::EQUAL);
}

TEST(TypeComparisonTest, Array) {
    types::Scalar float_type(types::PrimitiveType::Float);
    types::Array array_type1(float_type, symbolic::integer(10));
    types::Array array_type2(float_type, symbolic::integer(20));

    EXPECT_EQ(types::compare_types(array_type1, array_type2), types::TypeCompare::INCOMPATIBLE);

    types::Array array_type3(float_type, symbolic::integer(10));
    EXPECT_EQ(types::compare_types(array_type1, array_type3), types::TypeCompare::EQUAL);
}

TEST(TypeComparisonTest, Pointer) {
    types::Scalar float_type(types::PrimitiveType::Float);
    types::Pointer pointer_type1(float_type);
    types::Pointer pointer_type2(float_type);

    EXPECT_EQ(types::compare_types(pointer_type1, pointer_type2), types::TypeCompare::EQUAL);

    types::Scalar int_type(types::PrimitiveType::Int32);
    types::Pointer pointer_type3(int_type);
    EXPECT_EQ(types::compare_types(pointer_type1, pointer_type3), types::TypeCompare::INCOMPATIBLE);
}

TEST(TypeComparisonTest, Structure) {
    types::Structure structure_type1("test1");
    types::Structure structure_type2("test2");
    types::Structure structure_type3("test1");

    EXPECT_EQ(types::compare_types(structure_type1, structure_type2), types::TypeCompare::INCOMPATIBLE);
    EXPECT_EQ(types::compare_types(structure_type1, structure_type3), types::TypeCompare::EQUAL);
}

TEST(TypeComparisonTest, ArrayNesting) {
    types::Scalar float_type(types::PrimitiveType::Float);
    types::Array array_type1(float_type, symbolic::integer(10));
    types::Array array_type2(array_type1, symbolic::integer(20));

    types::Array array_type3(float_type, symbolic::integer(10));
    types::Array array_type4(array_type3, symbolic::integer(20));

    EXPECT_EQ(types::compare_types(array_type2, array_type4), types::TypeCompare::EQUAL);

    types::Array array_type5(float_type, symbolic::integer(15));
    types::Array array_type6(array_type5, symbolic::integer(20));

    EXPECT_EQ(types::compare_types(array_type2, array_type6), types::TypeCompare::INCOMPATIBLE);
}

TEST(TypeComparisonTest, PointerNesting) {
    types::Scalar float_type(types::PrimitiveType::Float);
    types::Pointer pointer_type1(float_type);
    types::Pointer pointer_type2(types::StorageType::CPU_Heap(), 8, "", pointer_type1);

    types::Scalar float_type2(types::PrimitiveType::Float);
    types::Pointer pointer_type3(float_type2);
    types::Pointer pointer_type4(types::StorageType::CPU_Heap(), 8, "", pointer_type3);

    EXPECT_EQ(types::compare_types(pointer_type2, pointer_type4), types::TypeCompare::EQUAL);

    types::Scalar int_type(types::PrimitiveType::Int32);
    types::Pointer pointer_type5(int_type);
    types::Pointer pointer_type6(types::StorageType::CPU_Heap(), 8, "", pointer_type5);

    EXPECT_EQ(types::compare_types(pointer_type2, pointer_type6), types::TypeCompare::INCOMPATIBLE);
}

TEST(TypeComparisonTest, ArrayPointerMix) {
    types::Scalar float_type(types::PrimitiveType::Float);
    types::Array array_type(float_type, symbolic::integer(10));
    types::Pointer pointer_type(float_type);

    EXPECT_EQ(types::compare_types(array_type, pointer_type), types::TypeCompare::INCOMPATIBLE);
}

TEST(TypeComparisonTest, UnequalArray) {
    types::Scalar float_type(types::PrimitiveType::Float);
    types::Array array_type1(float_type, symbolic::integer(10));

    EXPECT_EQ(types::compare_types(array_type1, float_type), types::TypeCompare::LARGER);
    EXPECT_EQ(types::compare_types(float_type, array_type1), types::TypeCompare::SMALLER);
}

TEST(TypeComparisonTest, UnequalPointer) {
    types::Scalar float_type(types::PrimitiveType::Float);
    types::Pointer pointer_type1(float_type);

    EXPECT_EQ(types::compare_types(pointer_type1, float_type), types::TypeCompare::LARGER);
    EXPECT_EQ(types::compare_types(float_type, pointer_type1), types::TypeCompare::SMALLER);
}

TEST(TypeComparisonTest, UnequalPointerArrayMix) {
    types::Scalar float_type(types::PrimitiveType::Float);
    types::Array array_type1(float_type, symbolic::integer(10));
    types::Pointer pointer_type1(array_type1);

    types::Pointer pointer_type2(float_type);


    EXPECT_EQ(types::compare_types(pointer_type1, pointer_type2), types::TypeCompare::LARGER);
    EXPECT_EQ(types::compare_types(pointer_type2, pointer_type1), types::TypeCompare::SMALLER);
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
