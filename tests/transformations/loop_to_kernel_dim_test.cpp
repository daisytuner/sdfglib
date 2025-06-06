#include "sdfg/transformations/loop_to_kernel_dim.h"

#include <gtest/gtest.h>

#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/element.h"
#include "sdfg/passes/structured_control_flow/dead_cfg_elimination.h"
#include "sdfg/passes/structured_control_flow/sequence_fusion.h"
#include "sdfg/schedule.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/pointer.h"

using namespace sdfg;

TEST(LoopToKernelDimTest, Basic) {
    /*
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType::NV_GLOBAL);

    auto& sdfg = builder.subject();
    auto& kernel = builder.add_kernel(
        sdfg.root(), sdfg.name(), DebugInfo(), symbolic::integer(1), symbolic::integer(1),
        symbolic::integer(1), symbolic::integer(32), symbolic::integer(8), symbolic::integer(1));
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("B", desc, true);

    types::Array array_desc(base_desc, {symbolic::integer(8)});
    types::Array array_desc2(array_desc, {symbolic::integer(32)});
    builder.add_container("B_shared", array_desc2);

    types::Scalar sym_desc(types::PrimitiveType::UInt32);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("i_shared", sym_desc);
    builder.add_container("i_access", sym_desc);

    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(8));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    auto indvar_shared = symbolic::symbol("i_shared");
    auto shared_bound =
        symbolic::And(symbolic::Lt(indvar_shared, symbolic::add(symbolic::integer(8), indvar)),
                      symbolic::Lt(indvar_shared, symbolic::max(symbolic::integer(0), bound)));
    auto shared_init = indvar;
    auto shared_update = symbolic::add(indvar_shared, symbolic::integer(1));

    auto& loop_shared =
        builder.add_for(body, indvar_shared, shared_bound, shared_init, shared_update);
    auto& body_shared = loop_shared.root();

    auto& block_shared = builder.add_block(body_shared);
    auto& B = builder.add_access(block_shared, "B");
    auto& B_shared = builder.add_access(block_shared, "B_shared");

    auto& tasklet_shared = builder.add_tasklet(block_shared, data_flow::TaskletCode::assign,
                                               {"_out", base_desc}, {{"_in", base_desc}});

    auto& shared_in = builder.add_memlet(
        block_shared, B, "void", tasklet_shared, "_in",
        {symbolic::add(symbolic::mul(symbolic::integer(512), indvar_shared),
                       symbolic::add(symbolic::threadIdx_x(),
                                     symbolic::mul(symbolic::blockDim_x(),
    symbolic::symbol("blockIdx.x"))))});

    auto& shared_out =
        builder.add_memlet(block_shared, tasklet_shared, "_out", B_shared, "void",
                           {symbolic::threadIdx_x(), symbolic::sub(indvar_shared, indvar)});

    auto& sync_block = builder.add_block(body);
    auto& libnode =
        builder.add_library_node(sync_block, data_flow::LibraryNodeCode::barrier_local, {}, {});

    auto indvar_access = symbolic::symbol("i_access");
    auto access_bound =
        symbolic::And(symbolic::Lt(indvar_access, symbolic::add(symbolic::integer(8), indvar)),
                      symbolic::Lt(indvar_access, symbolic::max(symbolic::integer(0), bound)));
    auto access_init = indvar;
    auto access_update = symbolic::add(indvar_access, symbolic::integer(1));

    auto& loop_access =
        builder.add_for(body, indvar_access, access_bound, access_init, access_update);
    auto& body_access = loop_access.root();

    auto& block_access = builder.add_block(body_access);
    auto& B_shared_access = builder.add_access(block_access, "B_shared");
    auto& A = builder.add_access(block_access, "A");

    auto& tasklet_access = builder.add_tasklet(block_access, data_flow::TaskletCode::assign,
                                               {"_out", base_desc}, {{"_in", base_desc}});

    auto& access_in =
        builder.add_memlet(block_access, B_shared_access, "void", tasklet_access, "_in",
                           {symbolic::threadIdx_x(), symbolic::sub(indvar_access, indvar)});

    auto& access_out = builder.add_memlet(
        block_access, tasklet_access, "_out", A, "void",
        {symbolic::add(symbolic::mul(symbolic::integer(512), indvar_access),
                       symbolic::add(symbolic::threadIdx_x(),
                                     symbolic::mul(symbolic::blockDim_x(),
    symbolic::symbol("blockIdx.x"))))});

    auto structured_sdfg = builder.move();

    auto schedule = std::make_unique<Schedule>(structured_sdfg);
    auto& analysis_manager = schedule->analysis_manager();
    auto& builder_opt = schedule->builder();

    // Apply
    transformations::LoopToKernelDim transformation(sdfg.root(), loop_shared);
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

    EXPECT_EQ(loop.root().size(), 3);

    // Check
    EXPECT_EQ(loop.indvar()->get_name(), "i");

    EXPECT_TRUE(dynamic_cast<structured_control_flow::IfElse*>(&loop.root().at(0).first) !=
                nullptr);
    auto shared_branch = static_cast<structured_control_flow::IfElse*>(&loop.root().at(0).first);

    EXPECT_EQ(shared_branch->size(), 1);
    EXPECT_TRUE(symbolic::eq(shared_branch->at(0).second,
                             symbolic::subs(loop_shared.condition(), loop_shared.indvar(),
                                            symbolic::add(loop.indvar(), kernel.threadIdx_y()))));

    EXPECT_TRUE(dynamic_cast<structured_control_flow::Block*>(
                    &shared_branch->at(0).first.at(0).first) != nullptr);
    auto shared_block =
        static_cast<structured_control_flow::Block*>(&shared_branch->at(0).first.at(0).first);
    EXPECT_EQ(shared_block->dataflow().nodes().size(), 3);
    EXPECT_EQ(shared_block->dataflow().edges().size(), 2);

    for (auto& node : shared_block->dataflow().edges()) {
        if (auto* access_node = dynamic_cast<data_flow::AccessNode*>(&node.src())) {
            EXPECT_EQ(access_node->data(), "B");
            EXPECT_EQ(node.subset().size(), 1);
            EXPECT_TRUE(symbolic::eq(
                node.subset().at(0),
                symbolic::add(
                    symbolic::mul(symbolic::integer(512),
                                  symbolic::add(kernel.threadIdx_y(), loop.indvar())),
                    symbolic::add(symbolic::threadIdx_x(),
                                  symbolic::mul(symbolic::blockDim_x(),
    symbolic::symbol("blockIdx.x")))))); } else if (auto* access_node =
    dynamic_cast<data_flow::AccessNode*>(&node.dst())) { EXPECT_EQ(access_node->data(), "B_shared");
            EXPECT_EQ(node.subset().size(), 2);
            EXPECT_TRUE(symbolic::eq(node.subset().at(0), symbolic::threadIdx_x()));
            EXPECT_TRUE(symbolic::eq(node.subset().at(1), kernel.threadIdx_y()));
        } else {
            FAIL();
        }
    }
    */
}

TEST(LoopToKernelDimTest, DimNotAvailable) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType::NV_GLOBAL);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("B", desc, true);

    types::Array array_desc(base_desc, {symbolic::integer(8)});
    types::Array array_desc2(array_desc, {symbolic::integer(32)});
    builder.add_container("B_shared", array_desc2);

    types::Scalar sym_desc(types::PrimitiveType::UInt32);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("i_shared", sym_desc);
    builder.add_container("i_access", sym_desc);

    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(8));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    auto indvar_shared = symbolic::symbol("i_shared");
    auto shared_bound =
        symbolic::And(symbolic::Lt(indvar_shared, symbolic::add(symbolic::integer(8), indvar)),
                      symbolic::Lt(indvar_shared, symbolic::max(symbolic::integer(0), bound)));
    auto shared_init = indvar;
    auto shared_update = symbolic::add(indvar_shared, symbolic::integer(1));

    auto& loop_shared =
        builder.add_for(body, indvar_shared, shared_bound, shared_init, shared_update);
    auto& body_shared = loop_shared.root();

    auto& block_shared = builder.add_block(body_shared);
    auto& B = builder.add_access(block_shared, "B");
    auto& B_shared = builder.add_access(block_shared, "B_shared");

    auto& tasklet_shared = builder.add_tasklet(block_shared, data_flow::TaskletCode::assign,
                                               {"_out", base_desc}, {{"_in", base_desc}});

    auto& shared_in = builder.add_memlet(
        block_shared, B, "void", tasklet_shared, "_in",
        {symbolic::add(
            symbolic::mul(symbolic::integer(512), indvar_shared),
            symbolic::add(symbolic::threadIdx_x(),
                          symbolic::mul(symbolic::blockDim_x(), symbolic::symbol("blockIdx.x"))))});

    auto& shared_out =
        builder.add_memlet(block_shared, tasklet_shared, "_out", B_shared, "void",
                           {symbolic::threadIdx_x(), symbolic::sub(indvar_shared, indvar)});

    auto& sync_block = builder.add_block(body);
    auto& libnode =
        builder.add_library_node(sync_block, data_flow::LibraryNodeCode::barrier_local, {}, {});

    auto indvar_access = symbolic::symbol("i_access");
    auto access_bound =
        symbolic::And(symbolic::Lt(indvar_access, symbolic::add(symbolic::integer(8), indvar)),
                      symbolic::Lt(indvar_access, symbolic::max(symbolic::integer(0), bound)));
    auto access_init = indvar;
    auto access_update = symbolic::add(indvar_access, symbolic::integer(1));

    auto& loop_access =
        builder.add_for(body, indvar_access, access_bound, access_init, access_update);
    auto& body_access = loop_access.root();

    auto& block_access = builder.add_block(body_access);
    auto& B_shared_access = builder.add_access(block_access, "B_shared");
    auto& A = builder.add_access(block_access, "A");

    auto& tasklet_access = builder.add_tasklet(block_access, data_flow::TaskletCode::assign,
                                               {"_out", base_desc}, {{"_in", base_desc}});

    auto& access_in =
        builder.add_memlet(block_access, B_shared_access, "void", tasklet_access, "_in",
                           {symbolic::threadIdx_x(), symbolic::sub(indvar_access, indvar)});

    auto& access_out = builder.add_memlet(
        block_access, tasklet_access, "_out", A, "void",
        {symbolic::add(
            symbolic::mul(symbolic::integer(512), indvar_access),
            symbolic::add(symbolic::threadIdx_x(),
                          symbolic::mul(symbolic::blockDim_x(), symbolic::symbol("blockIdx.x"))))});

    auto structured_sdfg = builder.move();

    auto schedule = std::make_unique<Schedule>(structured_sdfg);
    auto& analysis_manager = schedule->analysis_manager();
    auto& builder_opt = schedule->builder();

    // Apply
    transformations::LoopToKernelDim transformation(sdfg.root(), loop_shared);
    EXPECT_FALSE(transformation.can_be_applied(*schedule));
}

TEST(LoopToKernelDimTest, DimToSmall) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType::NV_GLOBAL);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("B", desc, true);

    types::Array array_desc(base_desc, {symbolic::integer(8)});
    types::Array array_desc2(array_desc, {symbolic::integer(32)});
    builder.add_container("B_shared", array_desc2);

    types::Scalar sym_desc(types::PrimitiveType::UInt32);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("i_shared", sym_desc);
    builder.add_container("i_access", sym_desc);

    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(8));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    auto indvar_shared = symbolic::symbol("i_shared");
    auto shared_bound =
        symbolic::And(symbolic::Lt(indvar_shared, symbolic::add(symbolic::integer(8), indvar)),
                      symbolic::Lt(indvar_shared, symbolic::max(symbolic::integer(0), bound)));
    auto shared_init = indvar;
    auto shared_update = symbolic::add(indvar_shared, symbolic::integer(1));

    auto& loop_shared =
        builder.add_for(body, indvar_shared, shared_bound, shared_init, shared_update);
    auto& body_shared = loop_shared.root();

    auto& block_shared = builder.add_block(body_shared);
    auto& B = builder.add_access(block_shared, "B");
    auto& B_shared = builder.add_access(block_shared, "B_shared");

    auto& tasklet_shared = builder.add_tasklet(block_shared, data_flow::TaskletCode::assign,
                                               {"_out", base_desc}, {{"_in", base_desc}});

    auto& shared_in = builder.add_memlet(
        block_shared, B, "void", tasklet_shared, "_in",
        {symbolic::add(
            symbolic::mul(symbolic::integer(512), indvar_shared),
            symbolic::add(symbolic::threadIdx_x(),
                          symbolic::mul(symbolic::blockDim_x(), symbolic::symbol("blockIdx.x"))))});

    auto& shared_out =
        builder.add_memlet(block_shared, tasklet_shared, "_out", B_shared, "void",
                           {symbolic::threadIdx_x(), symbolic::sub(indvar_shared, indvar)});

    auto& sync_block = builder.add_block(body);
    auto& libnode =
        builder.add_library_node(sync_block, data_flow::LibraryNodeCode::barrier_local, {}, {});

    auto indvar_access = symbolic::symbol("i_access");
    auto access_bound =
        symbolic::And(symbolic::Lt(indvar_access, symbolic::add(symbolic::integer(8), indvar)),
                      symbolic::Lt(indvar_access, symbolic::max(symbolic::integer(0), bound)));
    auto access_init = indvar;
    auto access_update = symbolic::add(indvar_access, symbolic::integer(1));

    auto& loop_access =
        builder.add_for(body, indvar_access, access_bound, access_init, access_update);
    auto& body_access = loop_access.root();

    auto& block_access = builder.add_block(body_access);
    auto& B_shared_access = builder.add_access(block_access, "B_shared");
    auto& A = builder.add_access(block_access, "A");

    auto& tasklet_access = builder.add_tasklet(block_access, data_flow::TaskletCode::assign,
                                               {"_out", base_desc}, {{"_in", base_desc}});

    auto& access_in =
        builder.add_memlet(block_access, B_shared_access, "void", tasklet_access, "_in",
                           {symbolic::threadIdx_x(), symbolic::sub(indvar_access, indvar)});

    auto& access_out = builder.add_memlet(
        block_access, tasklet_access, "_out", A, "void",
        {symbolic::add(
            symbolic::mul(symbolic::integer(512), indvar_access),
            symbolic::add(symbolic::threadIdx_x(),
                          symbolic::mul(symbolic::blockDim_x(), symbolic::symbol("blockIdx.x"))))});

    auto structured_sdfg = builder.move();

    auto schedule = std::make_unique<Schedule>(structured_sdfg);
    auto& analysis_manager = schedule->analysis_manager();
    auto& builder_opt = schedule->builder();

    // Apply
    transformations::LoopToKernelDim transformation(sdfg.root(), loop_shared);
    EXPECT_FALSE(transformation.can_be_applied(*schedule));
}

TEST(LoopToKernelDimTest, NonIndvarAccess) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType::NV_GLOBAL);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("B", desc, true);

    types::Array array_desc(base_desc, {symbolic::integer(8)});
    types::Array array_desc2(array_desc, {symbolic::integer(32)});
    builder.add_container("B_shared", array_desc2);

    types::Scalar sym_desc(types::PrimitiveType::UInt32);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("i_shared", sym_desc);
    builder.add_container("i_access", sym_desc);

    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(8));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    auto indvar_shared = symbolic::symbol("i_shared");
    auto shared_bound =
        symbolic::And(symbolic::Lt(indvar_shared, symbolic::add(symbolic::integer(8), indvar)),
                      symbolic::Lt(indvar_shared, symbolic::max(symbolic::integer(0), bound)));
    auto shared_init = indvar;
    auto shared_update = symbolic::add(indvar_shared, symbolic::integer(1));

    auto& loop_shared =
        builder.add_for(body, indvar_shared, shared_bound, shared_init, shared_update);
    auto& body_shared = loop_shared.root();

    auto& block_shared = builder.add_block(body_shared);
    auto& B = builder.add_access(block_shared, "B");
    auto& B_shared = builder.add_access(block_shared, "B_shared");

    auto& tasklet_shared = builder.add_tasklet(block_shared, data_flow::TaskletCode::assign,
                                               {"_out", base_desc}, {{"_in", base_desc}});

    auto& shared_in = builder.add_memlet(
        block_shared, B, "void", tasklet_shared, "_in",
        {symbolic::add(
            symbolic::mul(symbolic::integer(512), indvar_shared),
            symbolic::add(symbolic::threadIdx_x(),
                          symbolic::mul(symbolic::blockDim_x(), symbolic::symbol("blockIdx.x"))))});

    auto& shared_out = builder.add_memlet(block_shared, tasklet_shared, "_out", B_shared, "void",
                                          {symbolic::threadIdx_x(), indvar});

    auto& sync_block = builder.add_block(body);
    auto& libnode =
        builder.add_library_node(sync_block, data_flow::LibraryNodeCode::barrier_local, {}, {});

    auto indvar_access = symbolic::symbol("i_access");
    auto access_bound =
        symbolic::And(symbolic::Lt(indvar_access, symbolic::add(symbolic::integer(8), indvar)),
                      symbolic::Lt(indvar_access, symbolic::max(symbolic::integer(0), bound)));
    auto access_init = indvar;
    auto access_update = symbolic::add(indvar_access, symbolic::integer(1));

    auto& loop_access =
        builder.add_for(body, indvar_access, access_bound, access_init, access_update);
    auto& body_access = loop_access.root();

    auto& block_access = builder.add_block(body_access);
    auto& B_shared_access = builder.add_access(block_access, "B_shared");
    auto& A = builder.add_access(block_access, "A");

    auto& tasklet_access = builder.add_tasklet(block_access, data_flow::TaskletCode::assign,
                                               {"_out", base_desc}, {{"_in", base_desc}});

    auto& access_in =
        builder.add_memlet(block_access, B_shared_access, "void", tasklet_access, "_in",
                           {symbolic::threadIdx_x(), symbolic::sub(indvar_access, indvar)});

    auto& access_out = builder.add_memlet(
        block_access, tasklet_access, "_out", A, "void",
        {symbolic::add(
            symbolic::mul(symbolic::integer(512), indvar_access),
            symbolic::add(symbolic::threadIdx_x(),
                          symbolic::mul(symbolic::blockDim_x(), symbolic::symbol("blockIdx.x"))))});

    auto structured_sdfg = builder.move();

    auto schedule = std::make_unique<Schedule>(structured_sdfg);
    auto& analysis_manager = schedule->analysis_manager();
    auto& builder_opt = schedule->builder();

    // Apply
    transformations::LoopToKernelDim transformation(sdfg.root(), loop_shared);
    EXPECT_FALSE(transformation.can_be_applied(*schedule));
}
