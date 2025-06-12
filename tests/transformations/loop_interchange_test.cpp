#include "sdfg/transformations/loop_interchange.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/for.h"

using namespace sdfg;

TEST(LoopInterchangeTest, Map_2D_Tiled) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Pointer desc2(*desc.clone());
    builder.add_container("A", desc2, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("i_tile", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("j_tile", sym_desc);

    // Define loop 1
    auto bound1 = symbolic::symbol("N");
    auto indvar1 = symbolic::symbol("i");
    auto init1 = symbolic::integer(0);
    auto condition1 = symbolic::Lt(indvar1, bound1);
    auto update1 = symbolic::add(indvar1, symbolic::integer(32));

    auto& loop1 = builder.add_for(root, indvar1, condition1, init1, update1);
    auto& body1 = loop1.root();

    // Define loop 1 tile
    auto indvar1_tile = symbolic::symbol("i_tile");
    auto init1_tile = indvar1;
    auto condition1_tile =
        symbolic::And(symbolic::Lt(indvar1_tile, symbolic::add(indvar1, symbolic::integer(32))),
                      symbolic::Lt(indvar1_tile, bound1));
    auto update1_tile = symbolic::add(indvar1_tile, symbolic::integer(1));

    auto& loop1_tile =
        builder.add_for(body1, indvar1_tile, condition1_tile, init1_tile, update1_tile);
    auto& body1_tile = loop1_tile.root();

    // Define loop 2
    auto bound2 = symbolic::symbol("M");
    auto indvar2 = symbolic::symbol("j");
    auto init2 = symbolic::integer(0);
    auto condition2 = symbolic::Lt(indvar2, bound2);
    auto update2 = symbolic::add(indvar2, symbolic::integer(16));

    auto& loop2 = builder.add_for(body1_tile, indvar2, condition2, init2, update2);
    auto& body2 = loop2.root();

    // Define loop 2 tile
    auto indvar2_tile = symbolic::symbol("j_tile");
    auto init2_tile = indvar2;
    auto condition2_tile =
        symbolic::And(symbolic::Lt(indvar2_tile, symbolic::add(indvar2, symbolic::integer(16))),
                      symbolic::Lt(indvar2_tile, bound2));
    auto update2_tile = symbolic::add(indvar2_tile, symbolic::integer(1));

    auto& loop2_tile =
        builder.add_for(body2, indvar2_tile, condition2_tile, init2_tile, update2_tile);
    auto& body2_tile = loop2_tile.root();

    // Add computation
    auto& block = builder.add_block(body2_tile);
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", base_desc},
                                        {{"_in", base_desc}});
    builder.add_memlet(block, A_in, "void", tasklet, "_in",
                       {symbolic::symbol("i_tile"), symbolic::symbol("j_tile")});
    builder.add_memlet(block, tasklet, "_out", A_out, "void",
                       {symbolic::symbol("i_tile"), symbolic::symbol("j_tile")});

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    // Apply
    transformations::LoopInterchange transformation(body1, loop1_tile, loop2);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& new_sdfg = builder_opt.subject();

    EXPECT_EQ(new_sdfg.root().size(), 1);

    EXPECT_TRUE(dynamic_cast<structured_control_flow::For*>(&new_sdfg.root().at(0).first) !=
                nullptr);
    auto new_loop1 = dynamic_cast<structured_control_flow::For*>(&new_sdfg.root().at(0).first);
    EXPECT_EQ(new_loop1->indvar()->get_name(), "i");
    EXPECT_EQ(new_loop1->root().size(), 1);

    EXPECT_TRUE(dynamic_cast<structured_control_flow::For*>(&new_loop1->root().at(0).first) !=
                nullptr);
    auto new_loop1_tile =
        dynamic_cast<structured_control_flow::For*>(&new_loop1->root().at(0).first);
    EXPECT_EQ(new_loop1_tile->indvar()->get_name(), "j");
    EXPECT_EQ(new_loop1_tile->root().size(), 1);

    EXPECT_TRUE(dynamic_cast<structured_control_flow::For*>(&new_loop1_tile->root().at(0).first) !=
                nullptr);
    auto new_loop2 =
        dynamic_cast<structured_control_flow::For*>(&new_loop1_tile->root().at(0).first);
    EXPECT_EQ(new_loop2->indvar()->get_name(), "i_tile");
    EXPECT_EQ(new_loop2->root().size(), 1);

    EXPECT_TRUE(dynamic_cast<structured_control_flow::For*>(&new_loop2->root().at(0).first) !=
                nullptr);
    auto new_loop2_tile =
        dynamic_cast<structured_control_flow::For*>(&new_loop2->root().at(0).first);
    EXPECT_EQ(new_loop2_tile->indvar()->get_name(), "j_tile");
}

TEST(LoopInterchangeTest, Reduction_2D_Tiled) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("i_tile", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("j_tile", sym_desc);

    // Define loop 1
    auto bound1 = symbolic::symbol("N");
    auto indvar1 = symbolic::symbol("i");
    auto init1 = symbolic::integer(0);
    auto condition1 = symbolic::Lt(indvar1, bound1);
    auto update1 = symbolic::add(indvar1, symbolic::integer(32));

    auto& loop1 = builder.add_for(root, indvar1, condition1, init1, update1);
    auto& body1 = loop1.root();

    // Define loop 1 tile
    auto indvar1_tile = symbolic::symbol("i_tile");
    auto init1_tile = indvar1;
    auto condition1_tile =
        symbolic::And(symbolic::Lt(indvar1_tile, symbolic::add(indvar1, symbolic::integer(32))),
                      symbolic::Lt(indvar1_tile, bound1));
    auto update1_tile = symbolic::add(indvar1_tile, symbolic::integer(1));

    auto& loop1_tile =
        builder.add_for(body1, indvar1_tile, condition1_tile, init1_tile, update1_tile);
    auto& body1_tile = loop1_tile.root();

    // Define loop 2
    auto bound2 = symbolic::symbol("M");
    auto indvar2 = symbolic::symbol("j");
    auto init2 = symbolic::integer(0);
    auto condition2 = symbolic::Lt(indvar2, bound2);
    auto update2 = symbolic::add(indvar2, symbolic::integer(16));

    auto& loop2 = builder.add_for(body1_tile, indvar2, condition2, init2, update2);
    auto& body2 = loop2.root();

    // Define loop 2 tile
    auto indvar2_tile = symbolic::symbol("j_tile");
    auto init2_tile = indvar2;
    auto condition2_tile =
        symbolic::And(symbolic::Lt(indvar2_tile, symbolic::add(indvar2, symbolic::integer(16))),
                      symbolic::Lt(indvar2_tile, bound2));
    auto update2_tile = symbolic::add(indvar2_tile, symbolic::integer(1));

    auto& loop2_tile =
        builder.add_for(body2, indvar2_tile, condition2_tile, init2_tile, update2_tile);
    auto& body2_tile = loop2_tile.root();

    // Add computation
    auto& block = builder.add_block(body2_tile);
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::add, {"_out", base_desc},
                                        {{"_in", base_desc}, {"1", base_desc}});
    builder.add_memlet(block, A_in, "void", tasklet, "_in", {symbolic::symbol("i_tile")});
    builder.add_memlet(block, tasklet, "_out", A_out, "void", {symbolic::symbol("i_tile")});

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    // Apply
    transformations::LoopInterchange transformation(body1, loop1_tile, loop2);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& new_sdfg = builder_opt.subject();

    EXPECT_EQ(new_sdfg.root().size(), 1);

    EXPECT_TRUE(dynamic_cast<structured_control_flow::For*>(&new_sdfg.root().at(0).first) !=
                nullptr);
    auto new_loop1 = dynamic_cast<structured_control_flow::For*>(&new_sdfg.root().at(0).first);
    EXPECT_EQ(new_loop1->indvar()->get_name(), "i");
    EXPECT_EQ(new_loop1->root().size(), 1);

    EXPECT_TRUE(dynamic_cast<structured_control_flow::For*>(&new_loop1->root().at(0).first) !=
                nullptr);
    auto new_loop1_tile =
        dynamic_cast<structured_control_flow::For*>(&new_loop1->root().at(0).first);
    EXPECT_EQ(new_loop1_tile->indvar()->get_name(), "j");
    EXPECT_EQ(new_loop1_tile->root().size(), 1);

    EXPECT_TRUE(dynamic_cast<structured_control_flow::For*>(&new_loop1_tile->root().at(0).first) !=
                nullptr);
    auto new_loop2 =
        dynamic_cast<structured_control_flow::For*>(&new_loop1_tile->root().at(0).first);
    EXPECT_EQ(new_loop2->indvar()->get_name(), "i_tile");
    EXPECT_EQ(new_loop2->root().size(), 1);

    EXPECT_TRUE(dynamic_cast<structured_control_flow::For*>(&new_loop2->root().at(0).first) !=
                nullptr);
    auto new_loop2_tile =
        dynamic_cast<structured_control_flow::For*>(&new_loop2->root().at(0).first);
    EXPECT_EQ(new_loop2_tile->indvar()->get_name(), "j_tile");
}

/*
TEST(LoopInterchangeTest, Reduction_3D_Tiled) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Pointer desc2(*desc.clone());
    builder.add_container("A", desc2, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("K", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("i_tile", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("j_tile", sym_desc);
    builder.add_container("k", sym_desc);
    builder.add_container("k_tile", sym_desc);

    // Define loop 1
    auto bound1 = symbolic::symbol("N");
    auto indvar1 = symbolic::symbol("i");
    auto init1 = symbolic::integer(0);
    auto condition1 = symbolic::Lt(indvar1, bound1);
    auto update1 = symbolic::add(indvar1, symbolic::integer(32));

    auto& loop1 = builder.add_for(root, indvar1, condition1, init1, update1);
    auto& body1 = loop1.root();

    // Define loop 1 tile
    auto indvar1_tile = symbolic::symbol("i_tile");
    auto init1_tile = indvar1;
    auto condition1_tile =
        symbolic::And(symbolic::Lt(indvar1_tile, symbolic::add(indvar1, symbolic::integer(32))),
                      symbolic::Lt(indvar1_tile, bound1));
    auto update1_tile = symbolic::add(indvar1_tile, symbolic::integer(1));

    auto& loop1_tile =
        builder.add_for(body1, indvar1_tile, condition1_tile, init1_tile, update1_tile);
    auto& body1_tile = loop1_tile.root();

    // Define loop 2
    auto bound2 = symbolic::symbol("M");
    auto indvar2 = symbolic::symbol("j");
    auto init2 = symbolic::integer(0);
    auto condition2 = symbolic::Lt(indvar2, bound2);
    auto update2 = symbolic::add(indvar2, symbolic::integer(16));

    auto& loop2 = builder.add_for(body1_tile, indvar2, condition2, init2, update2);
    auto& body2 = loop2.root();

    // Define loop 2 tile
    auto indvar2_tile = symbolic::symbol("j_tile");
    auto init2_tile = indvar2;
    auto condition2_tile =
        symbolic::And(symbolic::Lt(indvar2_tile, symbolic::add(indvar2, symbolic::integer(16))),
                      symbolic::Lt(indvar2_tile, bound2));
    auto update2_tile = symbolic::add(indvar2_tile, symbolic::integer(1));

    auto& loop2_tile =
        builder.add_for(body2, indvar2_tile, condition2_tile, init2_tile, update2_tile);
    auto& body2_tile = loop2_tile.root();

    // Define loop 3
    auto bound3 = symbolic::symbol("K");
    auto indvar3 = symbolic::symbol("k");
    auto init3 = symbolic::integer(0);
    auto condition3 = symbolic::Lt(indvar3, bound3);
    auto update3 = symbolic::add(indvar3, symbolic::integer(8));

    auto& loop3 = builder.add_for(body2_tile, indvar3, condition3, init3, update3);
    auto& body3 = loop3.root();

    // Define loop 3 tile
    auto indvar3_tile = symbolic::symbol("k_tile");
    auto init3_tile = indvar3;
    auto condition3_tile =
        symbolic::And(symbolic::Lt(indvar3_tile, symbolic::add(indvar3, symbolic::integer(8))),
                      symbolic::Lt(indvar3_tile, bound3));
    auto update3_tile = symbolic::add(indvar3_tile, symbolic::integer(1));

    auto& loop3_tile =
        builder.add_for(body3, indvar3_tile, condition3_tile, init3_tile, update3_tile);
    auto& body3_tile = loop3_tile.root();

    // Add computation
    auto& block = builder.add_block(body3_tile);
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::add, {"_out", base_desc},
                                        {{"_in", base_desc}, {"1", base_desc}});
    builder.add_memlet(block, A_in, "void", tasklet, "_in",
                       {symbolic::symbol("i_tile"), symbolic::symbol("j_tile")});
    builder.add_memlet(block, tasklet, "_out", A_out, "void",
                       {symbolic::symbol("i_tile"), symbolic::symbol("j_tile")});

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    // Apply
    transformations::LoopInterchange transformation(loop2.root(), loop2_tile, loop3);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& new_sdfg = builder_opt.subject();
    EXPECT_EQ(new_sdfg.root().size(), 1);
    EXPECT_TRUE(dynamic_cast<structured_control_flow::For*>(&new_sdfg.root().at(0).first) !=
                nullptr);
    auto new_loop1 = dynamic_cast<structured_control_flow::For*>(&new_sdfg.root().at(0).first);
    EXPECT_EQ(new_loop1->indvar()->get_name(), "i");
    EXPECT_EQ(new_loop1->root().size(), 1);
    EXPECT_TRUE(dynamic_cast<structured_control_flow::For*>(&new_loop1->root().at(0).first) !=
                nullptr);
    auto new_loop1_tile =
        dynamic_cast<structured_control_flow::For*>(&new_loop1->root().at(0).first);
    EXPECT_EQ(new_loop1_tile->indvar()->get_name(), "i_tile");
    EXPECT_EQ(new_loop1_tile->root().size(), 1);
    EXPECT_TRUE(dynamic_cast<structured_control_flow::For*>(&new_loop1_tile->root().at(0).first) !=
                nullptr);
    auto new_loop2 =
        dynamic_cast<structured_control_flow::For*>(&new_loop1_tile->root().at(0).first);
    EXPECT_EQ(new_loop2->root().size(), 1);
    EXPECT_TRUE(dynamic_cast<structured_control_flow::For*>(&new_loop2->root().at(0).first) !=
                nullptr);
    auto new_loop2_tile =
        dynamic_cast<structured_control_flow::For*>(&new_loop2->root().at(0).first);

    transformations::LoopInterchange transformation2(loop1_tile.root(), *new_loop2,
                                                     *new_loop2_tile);
    EXPECT_TRUE(transformation2.can_be_applied(builder_opt, analysis_manager));
    transformation2.apply(builder_opt, analysis_manager);

    auto& new_sdfg2 = builder_opt.subject();
    EXPECT_EQ(new_sdfg2.root().size(), 1);
    EXPECT_TRUE(dynamic_cast<structured_control_flow::For*>(&new_sdfg2.root().at(0).first) !=
                nullptr);
    auto new_loop1_2 = dynamic_cast<structured_control_flow::For*>(&new_sdfg2.root().at(0).first);
    EXPECT_EQ(new_loop1_2->indvar()->get_name(), "i");
    EXPECT_EQ(new_loop1_2->root().size(), 1);
    EXPECT_TRUE(dynamic_cast<structured_control_flow::For*>(&new_loop1_2->root().at(0).first) !=
                nullptr);
    auto new_loop1_tile_2 =
        dynamic_cast<structured_control_flow::For*>(&new_loop1_2->root().at(0).first);
    EXPECT_EQ(new_loop1_tile_2->indvar()->get_name(), "i_tile");
    EXPECT_EQ(new_loop1_tile_2->root().size(), 1);
    EXPECT_TRUE(dynamic_cast<structured_control_flow::For*>(
                    &new_loop1_tile_2->root().at(0).first) != nullptr);
    auto new_loop2_2 =
        dynamic_cast<structured_control_flow::For*>(&new_loop1_tile_2->root().at(0).first);
    EXPECT_EQ(new_loop2_2->indvar()->get_name(), "k");
    EXPECT_EQ(new_loop2_2->root().size(), 1);
    EXPECT_TRUE(dynamic_cast<structured_control_flow::For*>(&new_loop2_2->root().at(0).first) !=
                nullptr);
    auto new_loop2_tile_2 =
        dynamic_cast<structured_control_flow::For*>(&new_loop2_2->root().at(0).first);
    EXPECT_EQ(new_loop2_tile_2->indvar()->get_name(), "j");
    EXPECT_EQ(new_loop2_tile_2->root().size(), 1);
    EXPECT_TRUE(dynamic_cast<structured_control_flow::For*>(
                    &new_loop2_tile_2->root().at(0).first) != nullptr);
    auto new_loop3_2 =
        dynamic_cast<structured_control_flow::For*>(&new_loop2_tile_2->root().at(0).first);
    EXPECT_EQ(new_loop3_2->indvar()->get_name(), "j_tile");
    EXPECT_EQ(new_loop3_2->root().size(), 1);
    EXPECT_TRUE(dynamic_cast<structured_control_flow::For*>(&new_loop3_2->root().at(0).first) !=
                nullptr);
    auto new_loop3_tile_2 =
        dynamic_cast<structured_control_flow::For*>(&new_loop3_2->root().at(0).first);
    EXPECT_EQ(new_loop3_tile_2->indvar()->get_name(), "k_tile");
}
*/
