#include "sdfg/passes/structured_control_flow/block_fusion.h"

#include <gtest/gtest.h>

#include <ostream>

#include "sdfg/builder/sdfg_builder.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/math.h"

using namespace sdfg;

TEST(BlockFusionTest, Computational_Chain) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar desc_element(types::PrimitiveType::Double);
    types::Array desc_array(desc_element, symbolic::integer(10));
    builder.add_container("A", desc_array);

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root, control_flow::Assignments{});

    auto& node1_1 = builder.add_access(block1, "A");
    auto& node2_1 = builder.add_access(block1, "A");
    auto& tasklet_1 = builder.add_tasklet(block1, data_flow::TaskletCode::fma, "_out", {"2", "_in", "1"});
    builder.add_computational_memlet(block1, node1_1, tasklet_1, "_in", {symbolic::integer(0)});
    builder.add_computational_memlet(block1, tasklet_1, "_out", node2_1, {symbolic::integer(0)});

    auto& block2 = builder.add_block(root, control_flow::Assignments{});

    auto& node1_2 = builder.add_access(block2, "A");
    auto& node2_2 = builder.add_access(block2, "A");
    auto& tasklet_2 = builder.add_tasklet(block2, data_flow::TaskletCode::fma, "_out", {"2", "_in", "1"});
    builder.add_computational_memlet(block2, node1_2, tasklet_2, "_in", {symbolic::integer(0)});
    builder.add_computational_memlet(block2, tasklet_2, "_out", node2_2, {symbolic::integer(0)});

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

    auto first_block = dynamic_cast<const structured_control_flow::Block*>(&sdfg->root().at(0).first);
    EXPECT_EQ(first_block->dataflow().nodes().size(), 5);
}

TEST(BlockFusionTest, SymbolUsedInSubset_Dataflow) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar desc_element(types::PrimitiveType::Int32);
    types::Array desc_array(desc_element, symbolic::integer(10));
    builder.add_container("A", desc_array);
    builder.add_container("i", desc_element);

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root, control_flow::Assignments{});

    auto& node1_1 = builder.add_access(block1, "i");
    auto& tasklet_1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign, "_out", {"0"});
    builder.add_computational_memlet(block1, tasklet_1, "_out", node1_1, {});

    auto& block2 = builder.add_block(root, control_flow::Assignments{});

    auto& node2_1 = builder.add_access(block2, "A");
    auto& tasklet_2 = builder.add_tasklet(block2, data_flow::TaskletCode::fma, "_out", {"1"});
    builder.add_computational_memlet(block2, tasklet_2, "_out", node2_1, {symbolic::symbol("i")});

    auto sdfg = builder.move();

    // Fusion
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::BlockFusionPass fusion_pass;
    EXPECT_FALSE(fusion_pass.run(builder_opt, analysis_manager));

    sdfg = builder_opt.move();
    EXPECT_EQ(sdfg->root().size(), 2);
}

TEST(BlockFusionTest, SymbolUsedWithSubset_Transition) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar desc_element(types::PrimitiveType::Int32);
    types::Array desc_array(desc_element, symbolic::integer(10));
    builder.add_container("A", desc_array);
    builder.add_container("i", desc_element);

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root, {{symbolic::symbol("i"), symbolic::integer(0)}});

    auto& block2 = builder.add_block(root, control_flow::Assignments{});

    auto& node2_1 = builder.add_access(block2, "A");
    auto& tasklet_2 = builder.add_tasklet(block2, data_flow::TaskletCode::fma, "_out", {"1"});
    builder.add_computational_memlet(block2, tasklet_2, "_out", node2_1, {symbolic::symbol("i")});

    auto sdfg = builder.move();

    // Fusion
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::BlockFusionPass fusion_pass;
    EXPECT_FALSE(fusion_pass.run(builder_opt, analysis_manager));

    sdfg = builder_opt.move();
    EXPECT_EQ(sdfg->root().size(), 2);
}

TEST(BlockFusionTest, Computational_IndependentSubgraphs) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar desc_element(types::PrimitiveType::Double);
    types::Array desc_array(desc_element, symbolic::integer(10));
    builder.add_container("A", desc_array);
    builder.add_container("B", desc_array);

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root, control_flow::Assignments{});

    auto& node1_1 = builder.add_access(block1, "A");
    auto& node2_1 = builder.add_access(block1, "A");
    auto& tasklet_1 = builder.add_tasklet(block1, data_flow::TaskletCode::fma, "_out", {"2", "_in", "1"});
    builder.add_computational_memlet(block1, node1_1, tasklet_1, "_in", {symbolic::integer(0)});
    builder.add_computational_memlet(block1, tasklet_1, "_out", node2_1, {symbolic::integer(0)});

    auto& block2 = builder.add_block(root, control_flow::Assignments{});

    auto& node1_2 = builder.add_access(block2, "B");
    auto& node2_2 = builder.add_access(block2, "B");
    auto& tasklet_2 = builder.add_tasklet(block2, data_flow::TaskletCode::fma, "_out", {"2", "_in", "1"});
    builder.add_computational_memlet(block2, node1_2, tasklet_2, "_in", {symbolic::integer(0)});
    builder.add_computational_memlet(block2, tasklet_2, "_out", node2_2, {symbolic::integer(0)});

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

    auto& dataflow = dynamic_cast<const structured_control_flow::Block*>(&sdfg->root().at(0).first)->dataflow();
    EXPECT_EQ(dataflow.nodes().size(), 6);
    EXPECT_EQ(dataflow.edges().size(), 4);
    EXPECT_EQ(dataflow.weakly_connected_components().first, 2);
}

// TEST(BlockFusionTest, Computational_LibraryNode_WithoutSideEffects) {
//     builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

//     types::Scalar desc(types::PrimitiveType::Double);
//     types::Array array_desc(desc, symbolic::integer(10));
//     types::Array array_desc_2(array_desc, symbolic::integer(20));

//     builder.add_container("input", array_desc_2);
//     builder.add_container("tmp", array_desc_2);
//     builder.add_container("output", array_desc_2);

//     auto& block_1 = builder.add_block(builder.subject().root());

//     auto& input_node = builder.add_access(block_1, "input");
//     auto& tmp_node_out = builder.add_access(block_1, "tmp");
//     auto& relu_node =
//         static_cast<math::ml::ReLUNode&>(builder.add_library_node<math::ml::ReLUNode>(block_1, DebugInfo()));
//     builder.add_computational_memlet(
//         block_1,
//         input_node,
//         relu_node,
//         "X",
//         {symbolic::integer(0), symbolic::integer(0)},
//         array_desc_2,
//         block_1.debug_info()
//     );
//     builder.add_computational_memlet(
//         block_1,
//         relu_node,
//         "Y",
//         tmp_node_out,
//         {symbolic::integer(0), symbolic::integer(0)},
//         array_desc_2,
//         block_1.debug_info()
//     );

//     auto& block_2 = builder.add_block(builder.subject().root());

//     auto& tmp_node_in = builder.add_access(block_2, "tmp");
//     auto& output_node = builder.add_access(block_2, "output");
//     auto& relu_node_2 =
//         static_cast<math::ml::ReLUNode&>(builder.add_library_node<math::ml::ReLUNode>(block_2, DebugInfo()));
//     builder.add_computational_memlet(
//         block_2,
//         tmp_node_in,
//         relu_node_2,
//         "X",
//         {symbolic::integer(0), symbolic::integer(0)},
//         array_desc_2,
//         block_2.debug_info()
//     );
//     builder.add_computational_memlet(
//         block_2,
//         relu_node_2,
//         "Y",
//         output_node,
//         {symbolic::integer(0), symbolic::integer(0)},
//         array_desc_2,
//         block_2.debug_info()
//     );

//     auto sdfg = builder.move();

//     // Fusion
//     builder::StructuredSDFGBuilder builder_opt(sdfg);
//     analysis::AnalysisManager analysis_manager(builder_opt.subject());
//     passes::BlockFusionPass fusion_pass;
//     fusion_pass.run(builder_opt, analysis_manager);

//     sdfg = builder_opt.move();
//     EXPECT_EQ(sdfg->root().size(), 1);

//     auto& dataflow = dynamic_cast<const structured_control_flow::Block*>(&sdfg->root().at(0).first)->dataflow();
//     EXPECT_EQ(dataflow.nodes().size(), 5);
//     EXPECT_EQ(dataflow.edges().size(), 4);
//     EXPECT_EQ(dataflow.weakly_connected_components().first, 1);
// }

TEST(BlockFusionTest, Reference) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar desc_element(types::PrimitiveType::Double);
    types::Pointer desc_pointer(desc_element);
    builder.add_container("A", desc_pointer);
    builder.add_container("B", desc_pointer);

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root, control_flow::Assignments{});

    auto& node1_1 = builder.add_access(block1, "A");
    auto& node2_1 = builder.add_access(block1, "B");
    builder.add_reference_memlet(block1, node1_1, node2_1, {symbolic::integer(0)}, desc_pointer);

    auto& block2 = builder.add_block(root, control_flow::Assignments{});

    auto& node1_2 = builder.add_access(block2, "B");
    auto& node2_2 = builder.add_access(block2, "B");
    auto& tasklet_2 = builder.add_tasklet(block2, data_flow::TaskletCode::fma, "_out", {"2", "_in", "1"});
    builder.add_computational_memlet(block2, node1_2, tasklet_2, "_in", {symbolic::integer(0)}, desc_pointer);
    builder.add_computational_memlet(block2, tasklet_2, "_out", node2_2, {symbolic::integer(0)}, desc_pointer);

    auto sdfg = builder.move();

    EXPECT_EQ(sdfg->name(), "sdfg_1");
    EXPECT_EQ(sdfg->root().size(), 2);

    // Fusion
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::BlockFusionPass fusion_pass;
    EXPECT_FALSE(fusion_pass.run(builder_opt, analysis_manager));

    sdfg = builder_opt.move();
    EXPECT_EQ(sdfg->root().size(), 2);
}

TEST(BlockFusionTest, Dereference) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Pointer opaque_ptr;
    builder.add_container("A", opaque_ptr);
    builder.add_container("B", opaque_ptr);

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root, control_flow::Assignments{});

    auto& node1_1 = builder.add_access(block1, "A");
    auto& node2_1 = builder.add_access(block1, "B");

    types::Pointer desc_ptr_2(static_cast<const types::IType&>(opaque_ptr));
    builder.add_dereference_memlet(block1, node1_1, node2_1, true, desc_ptr_2);

    auto& block2 = builder.add_block(root, control_flow::Assignments{});

    auto& node1_2 = builder.add_access(block2, "B");
    auto& node2_2 = builder.add_access(block2, "B");
    auto& tasklet_2 = builder.add_tasklet(block2, data_flow::TaskletCode::fma, "_out", {"2", "_in", "1"});

    types::Scalar desc_element(types::PrimitiveType::Double);
    types::Pointer desc_ptr(desc_element);
    builder.add_computational_memlet(block2, node1_2, tasklet_2, "_in", {symbolic::integer(0)}, desc_ptr);
    builder.add_computational_memlet(block2, tasklet_2, "_out", node2_2, {symbolic::integer(0)}, desc_ptr);

    auto sdfg = builder.move();

    EXPECT_EQ(sdfg->name(), "sdfg_1");
    EXPECT_EQ(sdfg->root().size(), 2);

    // Fusion
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::BlockFusionPass fusion_pass;
    EXPECT_FALSE(fusion_pass.run(builder_opt, analysis_manager));

    sdfg = builder_opt.move();
    EXPECT_EQ(sdfg->root().size(), 2);
}
