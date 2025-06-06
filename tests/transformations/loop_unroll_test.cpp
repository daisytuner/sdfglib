#include "sdfg/transformations/loop_unroll.h"

#include <gtest/gtest.h>

#include "sdfg/schedule.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"

using namespace sdfg;

TEST(LoopUnrollTest, Basic) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType::CPU);

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
    auto bound = symbolic::integer(4);
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& access_in = builder.add_access(block, "A");
    auto& access_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", base_desc},
                                        {{"_in", base_desc}});
    auto& memlet_in =
        builder.add_memlet(block, access_in, "void", tasklet, "_in", {{symbolic::symbol("i")}});
    auto& memlet_out =
        builder.add_memlet(block, tasklet, "_out", access_out, "void", {{symbolic::symbol("i")}});

    auto structured_sdfg = builder.move();

    auto schedule = std::make_unique<Schedule>(structured_sdfg);
    auto& builder_opt = schedule->builder();

    auto& new_root = builder_opt.subject().root();
    // Apply
    transformations::LoopUnroll transformation(new_root, loop);
    EXPECT_TRUE(transformation.can_be_applied(*schedule));
    transformation.apply(*schedule);

    // Check
    {
        EXPECT_EQ(new_root.size(), 4);
        auto new_branch = dynamic_cast<structured_control_flow::IfElse*>(&new_root.at(0).first);
        EXPECT_NE(new_branch, nullptr);
        EXPECT_EQ(new_branch->size(), 1);
        EXPECT_TRUE(
            symbolic::eq(new_branch->at(0).second,
                         symbolic::Lt(symbolic::add(symbolic::integer(0), symbolic::integer(0)),
                                      symbolic::integer(4))));
        auto& new_body = new_branch->at(0).first;
        EXPECT_EQ(new_body.size(), 1);
        auto new_seq = dynamic_cast<structured_control_flow::Sequence*>(&new_body.at(0).first);
        EXPECT_NE(new_seq, nullptr);
        EXPECT_EQ(new_seq->size(), 1);
        auto new_block = dynamic_cast<structured_control_flow::Block*>(&new_seq->at(0).first);
        EXPECT_NE(new_block, nullptr);
        EXPECT_EQ(new_block->dataflow().nodes().size(), 3);
        EXPECT_EQ(new_block->dataflow().edges().size(), 2);
    }
    {
        EXPECT_EQ(new_root.size(), 4);
        auto new_branch = dynamic_cast<structured_control_flow::IfElse*>(&new_root.at(1).first);
        EXPECT_NE(new_branch, nullptr);
        EXPECT_EQ(new_branch->size(), 1);
        EXPECT_TRUE(
            symbolic::eq(new_branch->at(0).second,
                         symbolic::Lt(symbolic::add(symbolic::integer(0), symbolic::integer(0)),
                                      symbolic::integer(4))));
        auto& new_body = new_branch->at(0).first;
        EXPECT_EQ(new_body.size(), 1);
        auto new_seq = dynamic_cast<structured_control_flow::Sequence*>(&new_body.at(0).first);
        EXPECT_NE(new_seq, nullptr);
        EXPECT_EQ(new_seq->size(), 1);
        auto new_block = dynamic_cast<structured_control_flow::Block*>(&new_seq->at(0).first);
        EXPECT_NE(new_block, nullptr);
        EXPECT_EQ(new_block->dataflow().nodes().size(), 3);
        EXPECT_EQ(new_block->dataflow().edges().size(), 2);
    }
    {
        EXPECT_EQ(new_root.size(), 4);
        auto new_branch = dynamic_cast<structured_control_flow::IfElse*>(&new_root.at(2).first);
        EXPECT_NE(new_branch, nullptr);
        EXPECT_EQ(new_branch->size(), 1);
        EXPECT_TRUE(
            symbolic::eq(new_branch->at(0).second,
                         symbolic::Lt(symbolic::add(symbolic::integer(0), symbolic::integer(0)),
                                      symbolic::integer(4))));
        auto& new_body = new_branch->at(0).first;
        EXPECT_EQ(new_body.size(), 1);
        auto new_seq = dynamic_cast<structured_control_flow::Sequence*>(&new_body.at(0).first);
        EXPECT_NE(new_seq, nullptr);
        EXPECT_EQ(new_seq->size(), 1);
        auto new_block = dynamic_cast<structured_control_flow::Block*>(&new_seq->at(0).first);
        EXPECT_NE(new_block, nullptr);
        EXPECT_EQ(new_block->dataflow().nodes().size(), 3);
        EXPECT_EQ(new_block->dataflow().edges().size(), 2);
    }
    {
        EXPECT_EQ(new_root.size(), 4);
        auto new_branch = dynamic_cast<structured_control_flow::IfElse*>(&new_root.at(3).first);
        EXPECT_NE(new_branch, nullptr);
        EXPECT_EQ(new_branch->size(), 1);
        EXPECT_TRUE(
            symbolic::eq(new_branch->at(0).second,
                         symbolic::Lt(symbolic::add(symbolic::integer(0), symbolic::integer(0)),
                                      symbolic::integer(4))));
        auto& new_body = new_branch->at(0).first;
        EXPECT_EQ(new_body.size(), 1);
        auto new_seq = dynamic_cast<structured_control_flow::Sequence*>(&new_body.at(0).first);
        EXPECT_NE(new_seq, nullptr);
        EXPECT_EQ(new_seq->size(), 1);
        auto new_block = dynamic_cast<structured_control_flow::Block*>(&new_seq->at(0).first);
        EXPECT_NE(new_block, nullptr);
        EXPECT_EQ(new_block->dataflow().nodes().size(), 3);
        EXPECT_EQ(new_block->dataflow().edges().size(), 2);
    }
}

TEST(LoopUnrollTest, FirstIterationFail) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType::CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Int32);
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
    auto& access_in = builder.add_access(block, "i");
    auto& access_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", base_desc},
                                        {{"_in", base_desc}});
    auto& memlet_in = builder.add_memlet(block, access_in, "void", tasklet, "_in", {});
    auto& memlet_out =
        builder.add_memlet(block, tasklet, "_out", access_out, "void", {symbolic::integer(0)});

    auto structured_sdfg = builder.move();

    auto schedule = std::make_unique<Schedule>(structured_sdfg);
    auto& analysis_manager = schedule->analysis_manager();
    auto& builder_opt = schedule->builder();

    // Apply
    transformations::LoopUnroll transformation(root, loop);
    EXPECT_FALSE(transformation.can_be_applied(*schedule));
}
