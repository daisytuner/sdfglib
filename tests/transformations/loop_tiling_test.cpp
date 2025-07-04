#include "sdfg/transformations/loop_tiling.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/passes/structured_control_flow/dead_cfg_elimination.h"
#include "sdfg/passes/structured_control_flow/sequence_fusion.h"

using namespace sdfg;

TEST(LoopTilingTest, Basic) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

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

    auto& orig_loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = orig_loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet =
        builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", base_desc}, {{"_in", base_desc}});
    builder.add_memlet(block, A_in, "void", tasklet, "_in", {symbolic::symbol("i")});
    builder.add_memlet(block, tasklet, "_out", A_out, "void", {symbolic::symbol("i")});

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    // Apply
    transformations::LoopTiling transformation(orig_loop, 32);
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
    EXPECT_TRUE(dynamic_cast<structured_control_flow::For*>(&sdfg_opt.root().at(0).first) != nullptr);
    auto& loop = static_cast<structured_control_flow::For&>(sdfg_opt.root().at(0).first);

    EXPECT_EQ(loop.root().size(), 1);

    // Check
    EXPECT_EQ(loop.indvar()->get_name(), "i_tile0");

    EXPECT_TRUE(dynamic_cast<structured_control_flow::For*>(&loop.root().at(0).first) != nullptr);
    auto inner_loop = static_cast<structured_control_flow::For*>(&loop.root().at(0).first);
    EXPECT_EQ(inner_loop->indvar()->get_name(), "i");

    auto& outer_update = loop.update();
    EXPECT_TRUE(symbolic::eq(outer_update, symbolic::add(loop.indvar(), symbolic::integer(32))));

    auto& inner_init = inner_loop->init();
    EXPECT_TRUE(symbolic::eq(inner_init, loop.indvar()));

    auto& inner_condition_tile = inner_loop->condition();
    EXPECT_TRUE(symbolic::
                    eq(inner_condition_tile,
                       symbolic::
                           And(symbolic::Lt(inner_loop->indvar(), symbolic::add(loop.indvar(), symbolic::integer(32))),
                               symbolic::Lt(inner_loop->indvar(), bound))));
    auto& inner_update = inner_loop->update();
    EXPECT_TRUE(symbolic::eq(inner_update, symbolic::add(inner_loop->indvar(), symbolic::integer(1))));

    EXPECT_EQ(inner_loop->root().size(), 1);
    EXPECT_TRUE(dynamic_cast<structured_control_flow::Block*>(&inner_loop->root().at(0).first) != nullptr);

    EXPECT_EQ(builder_opt.subject().exists("i_tile0"), true);
}
