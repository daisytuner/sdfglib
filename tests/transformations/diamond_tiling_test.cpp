#include "sdfg/transformations/diamond_tiling.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/passes/structured_control_flow/dead_cfg_elimination.h"
#include "sdfg/passes/structured_control_flow/sequence_fusion.h"

using namespace sdfg;

TEST(DiamondTilingTest, Basic2D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array desc_1(base_desc, symbolic::symbol("M"));
    types::Pointer desc_2(desc_1);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    // Define outer loop (typically time dimension)
    auto indvar_i = symbolic::symbol("i");
    auto& outer_loop = builder.add_for(
        root,
        indvar_i,
        symbolic::Lt(indvar_i, symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(indvar_i, symbolic::integer(1))
    );
    auto& outer_body = outer_loop.root();

    // Define inner loop (typically space dimension)
    auto indvar_j = symbolic::symbol("j");
    auto& inner_loop = builder.add_for(
        outer_body,
        indvar_j,
        symbolic::Lt(indvar_j, symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(indvar_j, symbolic::integer(1))
    );
    auto& inner_body = inner_loop.root();

    // Add computation
    auto& block = builder.add_block(inner_body);
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, A_in, tasklet, "_in", {indvar_i, indvar_j}, desc_2);
    builder.add_computational_memlet(block, tasklet, "_out", A_out, {indvar_i, indvar_j}, desc_2);

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    // Apply diamond tiling
    transformations::DiamondTiling transformation(outer_loop, inner_loop, 32, 8);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    // Cleanup
    bool applies = false;
    passes::SequenceFusion sequence_fusion;
    passes::DeadCFGElimination dead_cfg;
    do {
        applies = false;
        applies |= dead_cfg.run(builder_opt, analysis_manager);
        applies |= sequence_fusion.run(builder_opt, analysis_manager);
    } while (applies);

    auto& sdfg_opt = builder_opt.subject();
    EXPECT_EQ(sdfg_opt.root().size(), 1);
    
    // After diamond tiling, the structure should be:
    // Level 0: i_tile (tiled outer loop)
    //   Level 1: j_tile (tiled inner loop, interchanged with original i)
    //     Level 2: i (original outer loop, interchanged with j_tile)
    //       Level 3: j (original inner loop)
    
    // Check outermost loop (i_tile)
    auto* outermost_loop = dynamic_cast<structured_control_flow::For*>(&sdfg_opt.root().at(0).first);
    EXPECT_TRUE(outermost_loop != nullptr);
    EXPECT_EQ(outermost_loop->indvar()->get_name(), "i_tile0");
    EXPECT_EQ(outermost_loop->root().size(), 1);

    // Check second level loop (j_tile)
    auto* second_loop = dynamic_cast<structured_control_flow::For*>(&outermost_loop->root().at(0).first);
    EXPECT_TRUE(second_loop != nullptr);
    EXPECT_EQ(second_loop->indvar()->get_name(), "j_tile0");
    EXPECT_EQ(second_loop->root().size(), 1);

    // Check third level loop (i)
    auto* third_loop = dynamic_cast<structured_control_flow::For*>(&second_loop->root().at(0).first);
    EXPECT_TRUE(third_loop != nullptr);
    EXPECT_EQ(third_loop->indvar()->get_name(), "i");
    EXPECT_EQ(third_loop->root().size(), 1);

    // Check innermost loop (j)
    auto* innermost_loop = dynamic_cast<structured_control_flow::For*>(&third_loop->root().at(0).first);
    EXPECT_TRUE(innermost_loop != nullptr);
    EXPECT_EQ(innermost_loop->indvar()->get_name(), "j");
    EXPECT_EQ(innermost_loop->root().size(), 1);

    // Check that the block is still there
    EXPECT_TRUE(dynamic_cast<structured_control_flow::Block*>(&innermost_loop->root().at(0).first) != nullptr);

    // Verify tile sizes
    auto& outermost_update = outermost_loop->update();
    EXPECT_TRUE(symbolic::eq(outermost_update, symbolic::add(outermost_loop->indvar(), symbolic::integer(32))));

    auto& second_update = second_loop->update();
    EXPECT_TRUE(symbolic::eq(second_update, symbolic::add(second_loop->indvar(), symbolic::integer(8))));
}

TEST(DiamondTilingTest, Map2D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array desc_1(base_desc, symbolic::symbol("M"));
    types::Pointer desc_2(desc_1);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    // Define outer map
    auto indvar_i = symbolic::symbol("i");
    auto& outer_loop = builder.add_map(
        root,
        indvar_i,
        symbolic::Lt(indvar_i, symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(indvar_i, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& outer_body = outer_loop.root();

    // Define inner map
    auto indvar_j = symbolic::symbol("j");
    auto& inner_loop = builder.add_map(
        outer_body,
        indvar_j,
        symbolic::Lt(indvar_j, symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(indvar_j, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& inner_body = inner_loop.root();

    // Add computation
    auto& block = builder.add_block(inner_body);
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, A_in, tasklet, "_in", {indvar_i, indvar_j}, desc_2);
    builder.add_computational_memlet(block, tasklet, "_out", A_out, {indvar_i, indvar_j}, desc_2);

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    // Apply diamond tiling
    transformations::DiamondTiling transformation(outer_loop, inner_loop, 16, 4);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    // Cleanup
    bool applies = false;
    passes::SequenceFusion sequence_fusion;
    passes::DeadCFGElimination dead_cfg;
    do {
        applies = false;
        applies |= dead_cfg.run(builder_opt, analysis_manager);
        applies |= sequence_fusion.run(builder_opt, analysis_manager);
    } while (applies);

    auto& sdfg_opt = builder_opt.subject();
    EXPECT_EQ(sdfg_opt.root().size(), 1);
    
    // Verify loop structure exists with correct nesting
    auto* outermost_loop = dynamic_cast<structured_control_flow::Map*>(&sdfg_opt.root().at(0).first);
    EXPECT_TRUE(outermost_loop != nullptr);
    EXPECT_EQ(outermost_loop->root().size(), 1);
    
    auto* second_loop = dynamic_cast<structured_control_flow::Map*>(&outermost_loop->root().at(0).first);
    EXPECT_TRUE(second_loop != nullptr);
    EXPECT_EQ(second_loop->root().size(), 1);
}

TEST(DiamondTilingTest, WithTransition) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array desc_1(base_desc, symbolic::symbol("M"));
    types::Pointer desc_2(desc_1);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    // Define outer loop with transition
    auto indvar_i = symbolic::symbol("i");
    auto& outer_loop = builder.add_for(
        root,
        indvar_i,
        symbolic::Lt(indvar_i, symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(indvar_i, symbolic::integer(1)),
        {{indvar_i, symbolic::zero()}}
    );
    auto& outer_body = outer_loop.root();

    // Define inner loop
    auto indvar_j = symbolic::symbol("j");
    auto& inner_loop = builder.add_for(
        outer_body,
        indvar_j,
        symbolic::Lt(indvar_j, symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(indvar_j, symbolic::integer(1))
    );
    auto& inner_body = inner_loop.root();

    // Add computation
    auto& block = builder.add_block(inner_body);
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, A_in, tasklet, "_in", {indvar_i, indvar_j}, desc_2);
    builder.add_computational_memlet(block, tasklet, "_out", A_out, {indvar_i, indvar_j}, desc_2);

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    // Apply diamond tiling
    transformations::DiamondTiling transformation(outer_loop, inner_loop, 16, 4);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    // Cleanup
    bool applies = false;
    passes::SequenceFusion sequence_fusion;
    passes::DeadCFGElimination dead_cfg;
    do {
        applies = false;
        applies |= dead_cfg.run(builder_opt, analysis_manager);
        applies |= sequence_fusion.run(builder_opt, analysis_manager);
    } while (applies);

    auto& sdfg_opt = builder_opt.subject();
    EXPECT_EQ(sdfg_opt.root().size(), 1);
    
    // Check that transition is preserved at the outermost level
    EXPECT_EQ(sdfg_opt.root().at(0).second.assignments().size(), 1);
    EXPECT_TRUE(symbolic::eq(sdfg_opt.root().at(0).second.assignments().at(indvar_i), symbolic::zero()));
}

TEST(DiamondTilingTest, DependentLoops) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array desc_1(base_desc, symbolic::symbol("M"));
    types::Pointer desc_2(desc_1);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);

    // Define outer loop
    auto indvar_i = symbolic::symbol("i");
    auto& outer_loop = builder.add_for(
        root,
        indvar_i,
        symbolic::Lt(indvar_i, symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(indvar_i, symbolic::integer(1))
    );
    auto& outer_body = outer_loop.root();

    // Define inner loop that depends on outer loop induction variable
    auto indvar_j = symbolic::symbol("j");
    auto offset = symbolic::add(indvar_i, symbolic::integer(1));
    auto& inner_loop = builder.add_for(
        outer_body,
        indvar_j,
        symbolic::Lt(indvar_j, symbolic::sub(symbolic::symbol("M"), offset)),
        symbolic::integer(0),
        symbolic::add(indvar_j, symbolic::integer(1))
    );
    auto& inner_body = inner_loop.root();

    // Add computation
    auto& block = builder.add_block(inner_body);
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, A_in, tasklet, "_in", {indvar_i, symbolic::add(indvar_j, offset)}, desc_2);
    builder.add_computational_memlet(
        block, tasklet, "_out", A_out, {symbolic::add(indvar_j, offset), indvar_i}, desc_2
    );

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::DiamondTiling transformation(outer_loop, inner_loop, 16, 4);
    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));
}

TEST(DiamondTilingTest, OuterLoopHasMultipleChildren) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array desc_1(base_desc, symbolic::symbol("M"));
    types::Pointer desc_2(desc_1);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    // Define outer loop
    auto indvar_i = symbolic::symbol("i");
    auto& outer_loop = builder.add_for(
        root,
        indvar_i,
        symbolic::Lt(indvar_i, symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(indvar_i, symbolic::integer(1))
    );
    auto& outer_body = outer_loop.root();
    
    // Add a blocking element before the inner loop
    auto& blocker = builder.add_block(outer_body);

    // Define inner loop
    auto indvar_j = symbolic::symbol("j");
    auto& inner_loop = builder.add_for(
        outer_body,
        indvar_j,
        symbolic::Lt(indvar_j, symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(indvar_j, symbolic::integer(1))
    );
    auto& inner_body = inner_loop.root();

    // Add computation
    auto& block = builder.add_block(inner_body);
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "A");
    auto& i = builder.add_access(block, "i");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, A_in, tasklet, "_in1", {indvar_i, indvar_j}, desc_2);
    builder.add_computational_memlet(block, i, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", A_out, {indvar_i, indvar_j}, desc_2);

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::DiamondTiling transformation(outer_loop, inner_loop, 16, 4);
    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));
}

TEST(DiamondTilingTest, InvalidTileSize) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array desc_1(base_desc, symbolic::symbol("M"));
    types::Pointer desc_2(desc_1);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    // Define outer loop
    auto indvar_i = symbolic::symbol("i");
    auto& outer_loop = builder.add_for(
        root,
        indvar_i,
        symbolic::Lt(indvar_i, symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(indvar_i, symbolic::integer(1))
    );
    auto& outer_body = outer_loop.root();

    // Define inner loop
    auto indvar_j = symbolic::symbol("j");
    auto& inner_loop = builder.add_for(
        outer_body,
        indvar_j,
        symbolic::Lt(indvar_j, symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(indvar_j, symbolic::integer(1))
    );
    auto& inner_body = inner_loop.root();

    // Add computation
    auto& block = builder.add_block(inner_body);
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, A_in, tasklet, "_in", {indvar_i, indvar_j}, desc_2);
    builder.add_computational_memlet(block, tasklet, "_out", A_out, {indvar_i, indvar_j}, desc_2);

    analysis::AnalysisManager analysis_manager(builder.subject());
    
    // Test with tile size 1 (invalid)
    transformations::DiamondTiling transformation1(outer_loop, inner_loop, 1, 8);
    EXPECT_FALSE(transformation1.can_be_applied(builder, analysis_manager));
    
    transformations::DiamondTiling transformation2(outer_loop, inner_loop, 16, 1);
    EXPECT_FALSE(transformation2.can_be_applied(builder, analysis_manager));
    
    transformations::DiamondTiling transformation3(outer_loop, inner_loop, 0, 8);
    EXPECT_FALSE(transformation3.can_be_applied(builder, analysis_manager));
}
