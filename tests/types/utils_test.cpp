#include "sdfg/types/utils.h"

#include <gtest/gtest.h>

#include "sdfg/builder/sdfg_builder.h"

using namespace sdfg;

TEST(TypeInferenceTest, Identity) {
    builder::SDFGBuilder builder("test");
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
    builder::SDFGBuilder builder("test");
    auto& function = builder.subject();

    types::Scalar scalar_type(types::PrimitiveType::Int32);
    data_flow::Subset subset = {symbolic::integer(0)};
    auto& inferred = types::infer_type(function, scalar_type, subset);
    EXPECT_EQ(inferred, scalar_type);
}

TEST(TypeInferenceTest, ElementType) {
    builder::SDFGBuilder builder("test");
    auto& function = builder.subject();

    types::Scalar scalar_type(types::PrimitiveType::Int32);
    types::Array array_type(scalar_type, symbolic::integer(10));

    data_flow::Subset subset = {symbolic::integer(0)};
    auto& inferred = types::infer_type(function, array_type, subset);
    EXPECT_EQ(inferred, scalar_type);
}

TEST(TypeInferenceTest, PointeeType) {
    builder::SDFGBuilder builder("test");
    auto& function = builder.subject();

    types::Scalar scalar_type(types::PrimitiveType::Int32);
    types::Pointer pointer_type(scalar_type);

    data_flow::Subset subset = {symbolic::integer(0)};
    auto& inferred = types::infer_type(function, pointer_type, subset);
    EXPECT_EQ(inferred, scalar_type);
}

TEST(TypeInferenceTest, StructureMember) {
    builder::SDFGBuilder builder("test");

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
