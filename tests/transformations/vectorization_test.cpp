#include "sdfg/transformations/vectorization.h"

#include <gtest/gtest.h>

#include "sdfg/schedule.h"

using namespace sdfg;

TEST(VectorizationTest, Contiguous) {
    builder::StructuredSDFGBuilder builder("sdfg_test");

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", base_desc},
                                        {{"_in", base_desc}});
    builder.add_memlet(block, A_in, "void", tasklet, "_in1", {symbolic::symbol("i")});
    builder.add_memlet(block, tasklet, "_out", A_out, "void", {symbolic::symbol("i")});

    auto structured_sdfg = builder.move();

    // Schedule
    auto schedule = std::make_unique<Schedule>(structured_sdfg);
    auto& analysis_manager = schedule->analysis_manager();
    auto& builder_opt = schedule->builder();

    // Apply
    auto& parent = builder_opt.parent(loop);
    transformations::Vectorization transformation(parent, loop);
    EXPECT_TRUE(transformation.can_be_applied(*schedule));
}

TEST(VectorizationTest, Constant_Assignment) {
    builder::StructuredSDFGBuilder builder("sdfg_test");

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", base_desc},
                                        {{"_in", base_desc}});
    builder.add_memlet(block, A_in, "void", tasklet, "_in1", {symbolic::integer(0)});
    builder.add_memlet(block, tasklet, "_out", A_out, "void", {});

    auto structured_sdfg = builder.move();

    // Schedule
    auto schedule = std::make_unique<Schedule>(structured_sdfg);
    auto& analysis_manager = schedule->analysis_manager();
    auto& builder_opt = schedule->builder();

    // Apply
    auto& parent = builder_opt.parent(loop);
    transformations::Vectorization transformation(parent, loop);
    EXPECT_TRUE(transformation.can_be_applied(*schedule));
}

TEST(VectorizationTest, Constant_Reduction) {
    builder::StructuredSDFGBuilder builder("sdfg_test");

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("a", base_desc, true);
    builder.add_container("A", desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& A_in = builder.add_access(block, "A");
    auto& A_in2 = builder.add_access(block, "a");
    auto& A_out = builder.add_access(block, "a");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::add, {"_out", base_desc},
                                        {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block, A_in, "void", tasklet, "_in1", {indvar});
    builder.add_memlet(block, A_in2, "void", tasklet, "_in2", {});
    builder.add_memlet(block, tasklet, "_out", A_out, "void", {});

    auto structured_sdfg = builder.move();

    // Schedule
    auto schedule = std::make_unique<Schedule>(structured_sdfg);
    auto& analysis_manager = schedule->analysis_manager();
    auto& builder_opt = schedule->builder();

    // Apply
    auto& parent = builder_opt.parent(loop);
    transformations::Vectorization transformation(parent, loop);
    EXPECT_TRUE(transformation.can_be_applied(*schedule));
}

TEST(VectorizationTest, Indirection_Scatter) {
    builder::StructuredSDFGBuilder builder("sdfg_test");

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::UInt64);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("B", desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("b", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);

    // Define indirection
    auto& A = builder.add_access(block, "A");
    auto& b = builder.add_access(block, "b");
    auto& tasklet1 = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", base_desc},
                                         {{"_in", base_desc}});
    builder.add_memlet(block, A, "void", tasklet1, "_in", {indvar});
    builder.add_memlet(block, tasklet1, "_out", b, "void", {});

    auto& block1 = builder.add_block(body);

    auto& A1 = builder.add_access(block1, "A");
    auto& A2 = builder.add_access(block1, "B");
    auto& tasklet = builder.add_tasklet(block1, data_flow::TaskletCode::assign, {"_out", base_desc},
                                        {{"_in", base_desc}});
    builder.add_memlet(block1, A1, "void", tasklet, "_in", {indvar});
    builder.add_memlet(block1, tasklet, "_out", A2, "void", {symbolic::symbol("b")});

    auto structured_sdfg = builder.move();

    // Schedule
    auto schedule = std::make_unique<Schedule>(structured_sdfg);
    auto& analysis_manager = schedule->analysis_manager();
    auto& builder_opt = schedule->builder();

    // Apply
    auto& parent = builder_opt.parent(loop);
    transformations::Vectorization transformation(parent, loop);
    EXPECT_FALSE(transformation.can_be_applied(*schedule));
}

TEST(VectorizationTest, Indirection_Gather) {
    builder::StructuredSDFGBuilder builder("sdfg_test");

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::UInt64);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("B", desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("b", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);

    // Define indirection
    auto& A = builder.add_access(block, "A");
    auto& b = builder.add_access(block, "b");
    auto& tasklet1 = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", base_desc},
                                         {{"_in", base_desc}});
    builder.add_memlet(block, A, "void", tasklet1, "_in", {indvar});
    builder.add_memlet(block, tasklet1, "_out", b, "void", {});

    auto& block1 = builder.add_block(body);

    auto& A1 = builder.add_access(block1, "A");
    auto& A2 = builder.add_access(block1, "B");
    auto& tasklet = builder.add_tasklet(block1, data_flow::TaskletCode::assign, {"_out", base_desc},
                                        {{"_in", base_desc}});
    builder.add_memlet(block1, A1, "void", tasklet, "_in", {symbolic::symbol("b")});
    builder.add_memlet(block1, tasklet, "_out", A2, "void", {indvar});

    auto structured_sdfg = builder.move();

    // Schedule
    auto schedule = std::make_unique<Schedule>(structured_sdfg);
    auto& analysis_manager = schedule->analysis_manager();
    auto& builder_opt = schedule->builder();

    // Apply
    auto& parent = builder_opt.parent(loop);
    transformations::Vectorization transformation(parent, loop);
    EXPECT_TRUE(transformation.can_be_applied(*schedule));
}

TEST(VectorizationTest, Tiling) {
    builder::StructuredSDFGBuilder builder("sdfg_test");

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("i_tile", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i_tile");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(32));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    auto indvar_inner = symbolic::symbol("i");
    auto& inner_loop =
        builder.add_for(body, indvar_inner,
                        symbolic::And(symbolic::Lt(indvar_inner, symbolic::integer(32)),
                                      symbolic::Lt(indvar_inner, bound)),
                        indvar, symbolic::add(indvar_inner, symbolic::integer(1)));
    auto& inner_body = inner_loop.root();

    // Add computation
    auto& block = builder.add_block(inner_body);
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", base_desc},
                                        {{"_in", base_desc}});
    builder.add_memlet(block, A_in, "void", tasklet, "_in1", {symbolic::symbol("i")});
    builder.add_memlet(block, tasklet, "_out", A_out, "void", {symbolic::symbol("i")});

    auto structured_sdfg = builder.move();

    // Schedule
    auto schedule = std::make_unique<Schedule>(structured_sdfg);
    auto& analysis_manager = schedule->analysis_manager();
    auto& builder_opt = schedule->builder();

    // Apply
    transformations::Vectorization transformation(body, inner_loop);
    EXPECT_TRUE(transformation.can_be_applied(*schedule));
}
