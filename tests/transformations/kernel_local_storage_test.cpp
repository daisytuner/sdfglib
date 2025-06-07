#include "sdfg/transformations/kernel_local_storage.h"

#include <gtest/gtest.h>

#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/element.h"
#include "sdfg/passes/structured_control_flow/dead_cfg_elimination.h"
#include "sdfg/passes/structured_control_flow/sequence_fusion.h"
#include "sdfg/schedule.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/transformations/loop_tiling.h"
#include "sdfg/types/pointer.h"

using namespace sdfg;

TEST(KernelLocalStorageTest, Basic) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_NV_GLOBAL);

    auto& sdfg = builder.subject();

    // Add assumptions about grid and block dims
    sdfg.assumption(symbolic::gridDim_x()).lower_bound(symbolic::integer(1));
    sdfg.assumption(symbolic::gridDim_x()).upper_bound(symbolic::integer(1));
    sdfg.assumption(symbolic::gridDim_y()).lower_bound(symbolic::integer(1));
    sdfg.assumption(symbolic::gridDim_y()).upper_bound(symbolic::integer(1));
    sdfg.assumption(symbolic::gridDim_z()).lower_bound(symbolic::integer(1));
    sdfg.assumption(symbolic::gridDim_z()).upper_bound(symbolic::integer(1));

    sdfg.assumption(symbolic::blockDim_x()).lower_bound(symbolic::integer(32));
    sdfg.assumption(symbolic::blockDim_x()).upper_bound(symbolic::integer(32));
    sdfg.assumption(symbolic::blockDim_y()).lower_bound(symbolic::integer(8));
    sdfg.assumption(symbolic::blockDim_y()).upper_bound(symbolic::integer(8));
    sdfg.assumption(symbolic::blockDim_z()).lower_bound(symbolic::integer(2));
    sdfg.assumption(symbolic::blockDim_z()).upper_bound(symbolic::integer(2));

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

    auto& loop = builder.add_for(sdfg.root(), indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", base_desc},
                                        {{"_in", base_desc}});
    builder.add_memlet(block, A_in, "void", tasklet, "_in", {symbolic::symbol("i")});
    builder.add_memlet(block, tasklet, "_out", A_out, "void", {symbolic::symbol("i")});

    auto structured_sdfg = builder.move();

    auto schedule = std::make_unique<Schedule>(structured_sdfg);
    auto& analysis_manager = schedule->analysis_manager();
    auto& builder_opt = schedule->builder();

    // Apply
    transformations::LoopTiling transformation(sdfg.root(), loop, 32);
    EXPECT_TRUE(transformation.can_be_applied(*schedule));
    transformation.apply(*schedule);

    // Cleanup
    bool applies = false;
    passes::SequenceFusion sequence_fusion;
    passes::DeadCFGElimination dead_cfg;
    do {
        applies = false;
        applies |= dead_cfg.run(builder_opt, analysis_manager);
        applies |= sequence_fusion.run(builder_opt, analysis_manager);
    } while (applies);

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
    EXPECT_TRUE(symbolic::eq(
        inner_condition_tile,
        symbolic::And(
            symbolic::Lt(inner_loop->indvar(), symbolic::add(loop.indvar(), symbolic::integer(32))),
            symbolic::Lt(inner_loop->indvar(), bound))));
    auto& inner_update = inner_loop->update();
    EXPECT_TRUE(
        symbolic::eq(inner_update, symbolic::add(inner_loop->indvar(), symbolic::integer(1))));

    EXPECT_EQ(inner_loop->root().size(), 1);
    EXPECT_TRUE(dynamic_cast<structured_control_flow::Block*>(&inner_loop->root().at(0).first) !=
                nullptr);

    EXPECT_EQ(builder_opt.subject().exists("i_tile0"), true);

    sdfg::types::Scalar output_element_type(types::PrimitiveType::Float);
    sdfg::types::Pointer output_pointer_type(output_element_type);

    sdfg::types::Scalar input_element_type(types::PrimitiveType::Float);
    sdfg::types::Pointer input_pointer_type(output_element_type);

    builder_opt.add_container("x", input_pointer_type);
    builder_opt.add_container("y", output_pointer_type);

    auto& inner_block = builder_opt.add_block(inner_loop->root());

    auto& xread = builder_opt.add_access(inner_block, "x");
    auto& yread = builder_opt.add_access(inner_block, "y");
    auto& inner_tasklet =
        builder_opt.add_tasklet(inner_block, data_flow::TaskletCode::add, {"_out", base_desc},
                                {{"_in1", base_desc}, {"_in2", base_desc}});

    auto subset_x = symbolic::add(
        inner_loop->indvar(),
        symbolic::mul(
            symbolic::integer(512),
            symbolic::add(symbolic::threadIdx_x(),
                          symbolic::mul(symbolic::blockDim_x(), symbolic::symbol("blockIdx.x")))));
    auto subset_y = symbolic::add(
        symbolic::add(symbolic::threadIdx_y(),
                      symbolic::mul(symbolic::blockDim_y(), symbolic::symbol("blockIdx.y"))),
        symbolic::mul(
            symbolic::integer(512),
            symbolic::add(symbolic::threadIdx_x(),
                          symbolic::mul(symbolic::blockDim_x(), symbolic::symbol("blockIdx.x")))));

    builder_opt.add_memlet(inner_block, xread, "void", inner_tasklet, "_in1", {subset_x});
    builder_opt.add_memlet(inner_block, yread, "void", inner_tasklet, "_in2", {subset_y});

    auto& ywrite = builder_opt.add_access(inner_block, "y");

    builder_opt.add_memlet(inner_block, inner_tasklet, "_out", ywrite, "void", {subset_y});

    transformations::KernelLocalStorage transformation2(sdfg.root(), loop, *inner_loop, "x");
    EXPECT_TRUE(transformation2.can_be_applied(*schedule));
    transformation2.apply(*schedule);

    // Check
    EXPECT_EQ(loop.root().size(), 3);
    EXPECT_TRUE(dynamic_cast<structured_control_flow::For*>(&loop.root().at(0).first) != nullptr);
    auto sharedLoop = static_cast<structured_control_flow::For*>(&loop.root().at(0).first);
    EXPECT_EQ(sharedLoop->indvar()->get_name(), "__daisy_shared_indvar_x");

    EXPECT_TRUE(symbolic::eq(sharedLoop->init(), inner_loop->init()));
    EXPECT_TRUE(symbolic::eq(
        sharedLoop->condition(),
        symbolic::subs(inner_loop->condition(), inner_loop->indvar(), sharedLoop->indvar())));
    EXPECT_TRUE(symbolic::eq(
        sharedLoop->update(),
        symbolic::subs(inner_loop->update(), inner_loop->indvar(), sharedLoop->indvar())));

    EXPECT_EQ(builder_opt.subject().exists("__daisy_shared_indvar_x"), true);
    EXPECT_EQ(builder_opt.subject().exists("__daisy_share_x"), true);
    EXPECT_EQ(builder_opt.subject().exists("__daisy_share_wrapper_x"), true);

    EXPECT_TRUE(nullptr != dynamic_cast<structured_control_flow::Block*>(&loop.root().at(1).first));
    auto* sync_block = static_cast<structured_control_flow::Block*>(&loop.root().at(1).first);
    EXPECT_EQ(sync_block->dataflow().nodes().size(), 1);
    EXPECT_EQ(sync_block->dataflow().edges().size(), 0);
    EXPECT_TRUE(nullptr !=
                dynamic_cast<data_flow::LibraryNode*>(&(*sync_block->dataflow().nodes().begin())));
    auto* sync_node =
        static_cast<data_flow::LibraryNode*>(&(*sync_block->dataflow().nodes().begin()));
    EXPECT_EQ(sync_node->code(), data_flow::LibraryNodeCode::barrier_local);

    EXPECT_EQ(sharedLoop->root().size(), 1);
    EXPECT_TRUE(dynamic_cast<structured_control_flow::Block*>(&sharedLoop->root().at(0).first) !=
                nullptr);

    auto shared_block =
        static_cast<structured_control_flow::Block*>(&sharedLoop->root().at(0).first);
    auto& graph = shared_block->dataflow();

    EXPECT_EQ(graph.tasklets().size(), 1);

    auto tasklet_shared = *graph.tasklets().begin();

    EXPECT_EQ(tasklet_shared->inputs().size(), 1);
    EXPECT_EQ(tasklet_shared->output().first, "_out");
    EXPECT_EQ(tasklet_shared->output().second.primitive_type(), types::PrimitiveType::Float);

    auto& input = *(graph.in_edges(*tasklet_shared).begin());
    EXPECT_TRUE(dynamic_cast<data_flow::AccessNode*>(&input.src()) != nullptr);
    auto& in_access = static_cast<data_flow::AccessNode&>(input.src());
    EXPECT_EQ(in_access.data(), "x");

    auto& output = *(graph.out_edges(*tasklet_shared).begin());
    EXPECT_TRUE(dynamic_cast<data_flow::AccessNode*>(&output.dst()) != nullptr);
    auto& out_access = static_cast<data_flow::AccessNode&>(output.dst());
    EXPECT_EQ(out_access.data(), "__daisy_share_x");
}