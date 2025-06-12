#include "sdfg/passes/structured_control_flow/block_fusion.h"

#include <gtest/gtest.h>

#include <ostream>

#include "sdfg/builder/sdfg_builder.h"
#include "sdfg/builder/structured_sdfg_builder.h"

using namespace sdfg;

TEST(BlockFusionTest, Chain) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar desc_element(types::PrimitiveType::Double);
    types::Array desc_array(desc_element, symbolic::integer(10));
    builder.add_container("A", desc_array);

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root, control_flow::Assignments{});

    auto& node1_1 = builder.add_access(block1, "A");
    auto& node2_1 = builder.add_access(block1, "A");
    auto& tasklet_1 =
        builder.add_tasklet(block1, data_flow::TaskletCode::fma, {"_out", desc_element},
                            {{"2", desc_element}, {"_in", desc_element}, {"1", desc_element}});
    builder.add_memlet(block1, node1_1, "void", tasklet_1, "_in", {symbolic::integer(0)});
    builder.add_memlet(block1, tasklet_1, "_out", node2_1, "void", {symbolic::integer(0)});

    auto& block2 = builder.add_block(root, control_flow::Assignments{});

    auto& node1_2 = builder.add_access(block2, "A");
    auto& node2_2 = builder.add_access(block2, "A");
    auto& tasklet_2 =
        builder.add_tasklet(block2, data_flow::TaskletCode::fma, {"_out", desc_element},
                            {{"2", desc_element}, {"_in", desc_element}, {"1", desc_element}});
    builder.add_memlet(block2, node1_2, "void", tasklet_2, "_in", {symbolic::integer(0)});
    builder.add_memlet(block2, tasklet_2, "_out", node2_2, "void", {symbolic::integer(0)});

    auto sdfg = builder.move();

    EXPECT_EQ(sdfg->name(), "sdfg_1");
    EXPECT_EQ(sdfg->root().size(), 2);

    // Fusion
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::BlockFusionPass fusion_pass;
    fusion_pass.run(builder_opt, analysis_manager);

    sdfg = builder_opt.move();
    EXPECT_EQ(sdfg->root().size(), 1);

    auto first_block =
        dynamic_cast<const structured_control_flow::Block*>(&sdfg->root().at(0).first);
    EXPECT_EQ(first_block->dataflow().nodes().size(), 5);
}

TEST(BlockFusionTest, Independent) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar desc_element(types::PrimitiveType::Double);
    types::Array desc_array(desc_element, symbolic::integer(10));
    builder.add_container("A", desc_array);
    builder.add_container("B", desc_array);

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root, control_flow::Assignments{});

    auto& node1_1 = builder.add_access(block1, "A");
    auto& node2_1 = builder.add_access(block1, "A");
    auto& tasklet_1 =
        builder.add_tasklet(block1, data_flow::TaskletCode::fma, {"_out", desc_element},
                            {{"2", desc_element}, {"_in", desc_element}, {"1", desc_element}});
    builder.add_memlet(block1, node1_1, "void", tasklet_1, "_in", {symbolic::integer(0)});
    builder.add_memlet(block1, tasklet_1, "_out", node2_1, "void", {symbolic::integer(0)});

    auto& block2 = builder.add_block(root, control_flow::Assignments{});

    auto& node1_2 = builder.add_access(block2, "B");
    auto& node2_2 = builder.add_access(block2, "B");
    auto& tasklet_2 =
        builder.add_tasklet(block2, data_flow::TaskletCode::fma, {"_out", desc_element},
                            {{"2", desc_element}, {"_in", desc_element}, {"1", desc_element}});
    builder.add_memlet(block2, node1_2, "void", tasklet_2, "_in", {symbolic::integer(0)});
    builder.add_memlet(block2, tasklet_2, "_out", node2_2, "void", {symbolic::integer(0)});

    auto sdfg = builder.move();

    EXPECT_EQ(sdfg->name(), "sdfg_1");
    EXPECT_EQ(sdfg->root().size(), 2);

    // Fusion
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::BlockFusionPass fusion_pass;
    fusion_pass.run(builder_opt, analysis_manager);

    sdfg = builder_opt.move();
    EXPECT_EQ(sdfg->root().size(), 2);
}
