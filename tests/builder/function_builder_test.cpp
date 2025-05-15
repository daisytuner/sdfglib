#include <gtest/gtest.h>

#include "sdfg/builder/sdfg_builder.h"

using namespace sdfg;

TEST(FunctionBuilderTest, Empty) {
    builder::SDFGBuilder builder("sdfg_1");

    auto sdfg = builder.move();

    EXPECT_EQ(sdfg->name(), "sdfg_1");
    EXPECT_EQ(sdfg->containers().size(), 0);
    EXPECT_FALSE(sdfg->debug_info().has());
}

TEST(FunctionBuilderTest, AddTransient) {
    builder::SDFGBuilder builder("sdfg_1");

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
    builder::SDFGBuilder builder("sdfg_1");

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
    builder::SDFGBuilder builder("sdfg_1");

    auto& container =
        builder.add_container("i", types::Scalar(types::PrimitiveType::UInt64), false, true);

    auto sdfg = builder.move();

    EXPECT_EQ(sdfg->containers().size(), 1);
    EXPECT_EQ(sdfg->exists("i"), true);
    EXPECT_EQ(sdfg->type("i"), types::Scalar(types::PrimitiveType::UInt64));
    EXPECT_FALSE(sdfg->is_argument("i"));
    EXPECT_TRUE(sdfg->is_external("i"));
    EXPECT_FALSE(sdfg->is_transient("i"));
}

TEST(FunctionBuilderTest, RemoveTransient) {
    builder::SDFGBuilder builder("sdfg_1");

    auto& container = builder.add_container("i", types::Scalar(types::PrimitiveType::UInt64));

    builder.remove_container("i");

    auto sdfg = builder.move();

    EXPECT_EQ(sdfg->containers().size(), 0);
    EXPECT_EQ(sdfg->exists("i"), false);
}

TEST(FunctionBuilderTest, RemoveArgument) {
    builder::SDFGBuilder builder("sdfg_1");

    auto& container = builder.add_container("i", types::Scalar(types::PrimitiveType::UInt64), true);

    EXPECT_DEATH(builder.remove_container("i"), "");
}

TEST(FunctionBuilderTest, RemoveExternal) {
    builder::SDFGBuilder builder("sdfg_1");

    auto& container =
        builder.add_container("i", types::Scalar(types::PrimitiveType::UInt64), false, true);

    EXPECT_DEATH(builder.remove_container("i"), "");
}

TEST(FunctionBuilderTest, AddStructure) {
    builder::SDFGBuilder builder("sdfg_1");

    auto& structure = builder.add_structure("struct_1", false);

    auto sdfg = builder.move();

    EXPECT_EQ(sdfg->structures().size(), 1);
    EXPECT_EQ(sdfg->structure("struct_1").name(), "struct_1");
}

TEST(FunctionBuilderTest, MakeArrayTransient) {
    builder::SDFGBuilder builder("sdfg_1");

    auto& container = builder.add_container("i", types::Scalar(types::PrimitiveType::UInt64));

    builder.make_array("i", symbolic::integer(10));

    auto sdfg = builder.move();

    EXPECT_EQ(sdfg->containers().size(), 1);
    EXPECT_EQ(sdfg->type("i"),
              types::Array(types::Scalar(types::PrimitiveType::UInt64), symbolic::integer(10)));
}

TEST(FunctionBuilderTest, MakeArrayArgument) {
    builder::SDFGBuilder builder("sdfg_1");

    auto& container = builder.add_container("i", types::Scalar(types::PrimitiveType::UInt64), true);

    EXPECT_DEATH(builder.make_array("i", symbolic::integer(10)), "");
}

TEST(FunctionBuilderTest, FindNewName) {
    builder::SDFGBuilder builder("sdfg_1");

    auto& container = builder.add_container("i", types::Scalar(types::PrimitiveType::UInt64));

    auto new_name = builder.find_new_name("i");

    EXPECT_EQ(new_name, "i0");
}

TEST(FunctionBuilderTest, Assumptions) {
    builder::SDFGBuilder builder("sdfg_1");

    auto& container = builder.add_container("i", types::Scalar(types::PrimitiveType::UInt8));
    auto sdfg = builder.move();

    auto symbol = symbolic::symbol("i");
    EXPECT_EQ(sdfg->has_assumption(symbol), true);

    auto assumption = sdfg->assumption(symbol);
    EXPECT_EQ(assumption.symbol()->get_name(), symbol->get_name());
}
