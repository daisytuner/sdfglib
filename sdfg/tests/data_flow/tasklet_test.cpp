#include <gtest/gtest.h>

#include "sdfg/builder/sdfg_builder.h"
using namespace sdfg;

TEST(TaskletTest, Casts_Trivial) {
    builder::SDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::UInt8);
    types::Scalar desc2(types::PrimitiveType::UInt8);
    builder.add_container("i1", desc);
    builder.add_container("i2", desc2);

    auto& access_node_1 = builder.add_access(state, "i1");
    auto& access_node_2 = builder.add_access(state, "i2");
    auto& tasklet_1 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, access_node_1, tasklet_1, "_in", {});
    builder.add_computational_memlet(state, tasklet_1, "_out", access_node_2, {});

    EXPECT_TRUE(tasklet_1.is_assign());
    EXPECT_TRUE(tasklet_1.is_trivial(builder.subject()));
    EXPECT_FALSE(tasklet_1.is_cast(builder.subject()));
}

TEST(TaskletTest, Casts_Zext) {
    builder::SDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::UInt8);
    types::Scalar desc2(types::PrimitiveType::UInt32);
    builder.add_container("i1", desc);
    builder.add_container("i2", desc2);

    auto& access_node_1 = builder.add_access(state, "i1");
    auto& access_node_2 = builder.add_access(state, "i2");
    auto& tasklet_1 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, access_node_1, tasklet_1, "_in", {});
    builder.add_computational_memlet(state, tasklet_1, "_out", access_node_2, {});

    EXPECT_TRUE(tasklet_1.is_assign());
    EXPECT_TRUE(tasklet_1.is_cast(builder.subject()));
    EXPECT_TRUE(tasklet_1.is_zext(builder.subject()));
}

TEST(TaskletTest, Casts_Sext) {
    builder::SDFGBuilder builder("sdfg_2", FunctionType_CPU);

    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Int8);
    types::Scalar desc2(types::PrimitiveType::Int32);
    builder.add_container("i1", desc);
    builder.add_container("i2", desc2);

    auto& access_node_1 = builder.add_access(state, "i1");
    auto& access_node_2 = builder.add_access(state, "i2");
    auto& tasklet_1 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, access_node_1, tasklet_1, "_in", {});
    builder.add_computational_memlet(state, tasklet_1, "_out", access_node_2, {});

    EXPECT_TRUE(tasklet_1.is_assign());
    EXPECT_TRUE(tasklet_1.is_cast(builder.subject()));
    EXPECT_TRUE(tasklet_1.is_sext(builder.subject()));
}

TEST(TaskletTest, Casts_Trunc) {
    builder::SDFGBuilder builder("sdfg_3", FunctionType_CPU);

    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::UInt32);
    types::Scalar desc2(types::PrimitiveType::UInt8);
    builder.add_container("i1", desc);
    builder.add_container("i2", desc2);

    auto& access_node_1 = builder.add_access(state, "i1");
    auto& access_node_2 = builder.add_access(state, "i2");
    auto& tasklet_1 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, access_node_1, tasklet_1, "_in", {});
    builder.add_computational_memlet(state, tasklet_1, "_out", access_node_2, {});

    EXPECT_TRUE(tasklet_1.is_assign());
    EXPECT_TRUE(tasklet_1.is_cast(builder.subject()));
    EXPECT_TRUE(tasklet_1.is_trunc(builder.subject()));
}

TEST(TaskletTest, Casts_Fptoui) {
    builder::SDFGBuilder builder("sdfg_4", FunctionType_CPU);

    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Float);
    types::Scalar desc2(types::PrimitiveType::UInt32);
    builder.add_container("i1", desc);
    builder.add_container("i2", desc2);

    auto& access_node_1 = builder.add_access(state, "i1");
    auto& access_node_2 = builder.add_access(state, "i2");
    auto& tasklet_1 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, access_node_1, tasklet_1, "_in", {});
    builder.add_computational_memlet(state, tasklet_1, "_out", access_node_2, {});

    EXPECT_TRUE(tasklet_1.is_assign());
    EXPECT_TRUE(tasklet_1.is_cast(builder.subject()));
    EXPECT_TRUE(tasklet_1.is_fptoui(builder.subject()));
}

TEST(TaskletTest, Casts_Fptosi) {
    builder::SDFGBuilder builder("sdfg_5", FunctionType_CPU);

    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Float);
    types::Scalar desc2(types::PrimitiveType::Int32);
    builder.add_container("i1", desc);
    builder.add_container("i2", desc2);

    auto& access_node_1 = builder.add_access(state, "i1");
    auto& access_node_2 = builder.add_access(state, "i2");
    auto& tasklet_1 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, access_node_1, tasklet_1, "_in", {});
    builder.add_computational_memlet(state, tasklet_1, "_out", access_node_2, {});

    EXPECT_TRUE(tasklet_1.is_assign());
    EXPECT_TRUE(tasklet_1.is_cast(builder.subject()));
    EXPECT_TRUE(tasklet_1.is_fptosi(builder.subject()));
}

TEST(TaskletTest, Casts_Uitofp) {
    builder::SDFGBuilder builder("sdfg_6", FunctionType_CPU);

    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::UInt32);
    types::Scalar desc2(types::PrimitiveType::Float);
    builder.add_container("i1", desc);
    builder.add_container("i2", desc2);

    auto& access_node_1 = builder.add_access(state, "i1");
    auto& access_node_2 = builder.add_access(state, "i2");
    auto& tasklet_1 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, access_node_1, tasklet_1, "_in", {});
    builder.add_computational_memlet(state, tasklet_1, "_out", access_node_2, {});

    EXPECT_TRUE(tasklet_1.is_assign());
    EXPECT_TRUE(tasklet_1.is_cast(builder.subject()));
    EXPECT_TRUE(tasklet_1.is_uitofp(builder.subject()));
}

TEST(TaskletTest, Casts_Sitofp) {
    builder::SDFGBuilder builder("sdfg_7", FunctionType_CPU);

    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Int32);
    types::Scalar desc2(types::PrimitiveType::Float);
    builder.add_container("i1", desc);
    builder.add_container("i2", desc2);

    auto& access_node_1 = builder.add_access(state, "i1");
    auto& access_node_2 = builder.add_access(state, "i2");
    auto& tasklet_1 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, access_node_1, tasklet_1, "_in", {});
    builder.add_computational_memlet(state, tasklet_1, "_out", access_node_2, {});

    EXPECT_TRUE(tasklet_1.is_assign());
    EXPECT_TRUE(tasklet_1.is_cast(builder.subject()));
    EXPECT_TRUE(tasklet_1.is_sitofp(builder.subject()));
}

TEST(TaskletTest, Casts_Fpext) {
    builder::SDFGBuilder builder("sdfg_8", FunctionType_CPU);

    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Float);
    types::Scalar desc2(types::PrimitiveType::Double);
    builder.add_container("i1", desc);
    builder.add_container("i2", desc2);

    auto& access_node_1 = builder.add_access(state, "i1");
    auto& access_node_2 = builder.add_access(state, "i2");
    auto& tasklet_1 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, access_node_1, tasklet_1, "_in", {});
    builder.add_computational_memlet(state, tasklet_1, "_out", access_node_2, {});

    EXPECT_TRUE(tasklet_1.is_assign());
    EXPECT_TRUE(tasklet_1.is_cast(builder.subject()));
    EXPECT_TRUE(tasklet_1.is_fpext(builder.subject()));
}

TEST(TaskletTest, Casts_Fptrunc) {
    builder::SDFGBuilder builder("sdfg_9", FunctionType_CPU);

    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Double);
    types::Scalar desc2(types::PrimitiveType::Float);
    builder.add_container("i1", desc);
    builder.add_container("i2", desc2);

    auto& access_node_1 = builder.add_access(state, "i1");
    auto& access_node_2 = builder.add_access(state, "i2");
    auto& tasklet_1 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, access_node_1, tasklet_1, "_in", {});
    builder.add_computational_memlet(state, tasklet_1, "_out", access_node_2, {});

    EXPECT_TRUE(tasklet_1.is_assign());
    EXPECT_TRUE(tasklet_1.is_cast(builder.subject()));
    EXPECT_TRUE(tasklet_1.is_fptrunc(builder.subject()));
}
