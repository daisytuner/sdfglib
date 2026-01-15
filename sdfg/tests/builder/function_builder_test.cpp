#include <gtest/gtest.h>

#include "sdfg/builder/sdfg_builder.h"

using namespace sdfg;

TEST(FunctionBuilderTest, Empty) {
    builder::SDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto sdfg = builder.move();

    EXPECT_EQ(sdfg->name(), "sdfg_1");
    EXPECT_EQ(sdfg->containers().size(), 0);
}

TEST(FunctionBuilderTest, AddTransient) {
    builder::SDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& container = builder.add_container("i", types::Scalar(types::PrimitiveType::UInt64));

    auto sdfg = builder.move();

    EXPECT_EQ(sdfg->containers().size(), 1);
    EXPECT_EQ(sdfg->exists("i"), true);
    EXPECT_EQ(sdfg->type("i"), types::Scalar(types::PrimitiveType::UInt64));
    EXPECT_FALSE(sdfg->is_argument("i"));
    EXPECT_FALSE(sdfg->is_external("i"));
    EXPECT_TRUE(sdfg->is_transient("i"));
}

TEST(FunctionBuilderTest, AddArgument) {
    builder::SDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& container = builder.add_container("i", types::Scalar(types::PrimitiveType::UInt64), true);

    auto sdfg = builder.move();

    EXPECT_EQ(sdfg->containers().size(), 1);
    EXPECT_EQ(sdfg->exists("i"), true);
    EXPECT_EQ(sdfg->type("i"), types::Scalar(types::PrimitiveType::UInt64));
    EXPECT_TRUE(sdfg->is_argument("i"));
    EXPECT_FALSE(sdfg->is_external("i"));
    EXPECT_FALSE(sdfg->is_transient("i"));
}

TEST(FunctionBuilderTest, AddExternal) {
    builder::SDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& container = builder.add_container("i", types::Scalar(types::PrimitiveType::UInt64), false, true);

    auto sdfg = builder.move();

    EXPECT_EQ(sdfg->containers().size(), 1);
    EXPECT_EQ(sdfg->exists("i"), true);
    EXPECT_EQ(sdfg->type("i"), types::Scalar(types::PrimitiveType::UInt64));
    EXPECT_FALSE(sdfg->is_argument("i"));
    EXPECT_TRUE(sdfg->is_external("i"));
    EXPECT_FALSE(sdfg->is_transient("i"));
}

TEST(FunctionBuilderTest, AddExternal_LinkageType_External) {
    builder::SDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& container = builder.add_external("i", types::Scalar(types::PrimitiveType::UInt64), LinkageType_External);

    auto sdfg = builder.move();

    EXPECT_EQ(sdfg->containers().size(), 1);
    EXPECT_EQ(sdfg->exists("i"), true);
    EXPECT_EQ(sdfg->type("i"), types::Scalar(types::PrimitiveType::UInt64));
    EXPECT_FALSE(sdfg->is_argument("i"));
    EXPECT_TRUE(sdfg->is_external("i"));
    EXPECT_FALSE(sdfg->is_transient("i"));
    EXPECT_EQ(sdfg->linkage_type("i"), LinkageType_External);
}

TEST(FunctionBuilderTest, AddExternal_LinkageType_Internal) {
    builder::SDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& container = builder.add_external("i", types::Scalar(types::PrimitiveType::UInt64), LinkageType_Internal);

    auto sdfg = builder.move();

    EXPECT_EQ(sdfg->containers().size(), 1);
    EXPECT_EQ(sdfg->exists("i"), true);
    EXPECT_EQ(sdfg->type("i"), types::Scalar(types::PrimitiveType::UInt64));
    EXPECT_FALSE(sdfg->is_argument("i"));
    EXPECT_TRUE(sdfg->is_external("i"));
    EXPECT_FALSE(sdfg->is_transient("i"));
    EXPECT_EQ(sdfg->linkage_type("i"), LinkageType_Internal);
}

TEST(FunctionBuilderTest, RemoveTransient) {
    builder::SDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& container = builder.add_container("i", types::Scalar(types::PrimitiveType::UInt64));

    builder.remove_container("i");

    auto sdfg = builder.move();

    EXPECT_EQ(sdfg->containers().size(), 0);
    EXPECT_EQ(sdfg->exists("i"), false);
}

TEST(FunctionBuilderTest, RemoveArgument) {
    builder::SDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& container = builder.add_container("i", types::Scalar(types::PrimitiveType::UInt64), true);

    EXPECT_THROW(builder.remove_container("i"), sdfg::InvalidSDFGException);
}

TEST(FunctionBuilderTest, RemoveExternal) {
    builder::SDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& container = builder.add_container("i", types::Scalar(types::PrimitiveType::UInt64), false, true);

    EXPECT_THROW(builder.remove_container("i"), sdfg::InvalidSDFGException);
}

TEST(FunctionBuilderTest, AddStructure) {
    builder::SDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& structure = builder.add_structure("struct_1", false);

    auto sdfg = builder.move();

    EXPECT_EQ(sdfg->structures().size(), 1);
    EXPECT_EQ(sdfg->structure("struct_1").name(), "struct_1");
}

TEST(FunctionBuilderTest, FindNewName) {
    builder::SDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& container = builder.add_container("i", types::Scalar(types::PrimitiveType::UInt64));

    auto new_name = builder.find_new_name("i");

    EXPECT_EQ(new_name, "i0");
}

TEST(FunctionBuilderTest, Assumptions) {
    builder::SDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& container = builder.add_container("i", types::Scalar(types::PrimitiveType::UInt8));
    auto sdfg = builder.move();

    auto symbol = symbolic::symbol("i");
    EXPECT_EQ(sdfg->has_assumption(symbol), true);

    auto assumption = sdfg->assumption(symbol);
    EXPECT_EQ(assumption.symbol()->get_name(), symbol->get_name());
}
