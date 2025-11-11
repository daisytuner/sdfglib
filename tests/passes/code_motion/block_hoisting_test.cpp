#include "sdfg/passes/code_motion/block_hoisting.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/function.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"

using namespace sdfg;

TEST(BlockHoistingTest, Map_InvariantMove) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    types::Pointer opaque_ptr;
    builder.add_container("A", opaque_ptr, true);
    builder.add_container("a", opaque_ptr);
    builder.add_container("B", opaque_ptr, true);
    builder.add_container("i", sym_desc);

    auto& map_stmt = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::zero(),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& body = map_stmt.root();

    // Loop invariant move
    auto& block1 = builder.add_block(body);
    {
        auto& a = builder.add_access(block1, "A");
        auto& a_ = builder.add_access(block1, "a");
        builder.add_dereference_memlet(block1, a, a_, true, types::Pointer(static_cast<const types::IType&>(opaque_ptr)));
    }

    // Loop variant computation
    auto& block2 = builder.add_block(body);
    {
        auto& a = builder.add_access(block2, "a");
        auto& b = builder.add_access(block2, "B");
        auto& tasklet = builder.add_tasklet(block2, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
        builder.add_computational_memlet(block2, a, tasklet, "_in", {symbolic::symbol("i")}, desc_ptr);
        builder.add_computational_memlet(block2, tasklet, "_out", b, {symbolic::symbol("i")}, desc_ptr);
    }

    // Apply pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::BlockHoistingPass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    EXPECT_EQ(root.size(), 2);
    EXPECT_EQ(body.size(), 1);
}

TEST(BlockHoistingTest, Map_InvariantView) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    types::Pointer opaque_ptr;
    builder.add_container("A", opaque_ptr, true);
    builder.add_container("a", opaque_ptr);
    builder.add_container("B", opaque_ptr, true);
    builder.add_container("i", sym_desc);

    auto& map_stmt = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::zero(),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& body = map_stmt.root();

    // Loop invariant view
    auto& block1 = builder.add_block(body);
    {
        auto& a = builder.add_access(block1, "A");
        auto& a_ = builder.add_access(block1, "a");
        builder.add_reference_memlet(block1, a, a_, {symbolic::one()}, desc_ptr);
    }

    // Loop variant computation
    auto& block2 = builder.add_block(body);
    {
        auto& a = builder.add_access(block2, "a");
        auto& b = builder.add_access(block2, "B");
        auto& tasklet = builder.add_tasklet(block2, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
        builder.add_computational_memlet(block2, a, tasklet, "_in", {symbolic::symbol("i")}, desc_ptr);
        builder.add_computational_memlet(block2, tasklet, "_out", b, {symbolic::symbol("i")}, desc_ptr);
    }

    // Apply pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::BlockHoistingPass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    EXPECT_EQ(root.size(), 2);
    EXPECT_EQ(body.size(), 1);
}

TEST(BlockHoistingTest, For_InvariantMove) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    types::Pointer opaque_ptr;
    builder.add_container("A", opaque_ptr, true);
    builder.add_container("a", opaque_ptr);
    builder.add_container("B", opaque_ptr, true);
    builder.add_container("i", sym_desc);

    auto& for_stmt = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::zero(),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );
    auto& body = for_stmt.root();

    // Loop invariant move
    auto& block1 = builder.add_block(body);
    {
        auto& a = builder.add_access(block1, "A");
        auto& a_ = builder.add_access(block1, "a");
        builder.add_dereference_memlet(block1, a, a_, true, types::Pointer(static_cast<const types::IType&>(opaque_ptr)));
    }

    // Loop variant computation
    auto& block2 = builder.add_block(body);
    {
        auto& a = builder.add_access(block2, "a");
        auto& b = builder.add_access(block2, "B");
        auto& tasklet = builder.add_tasklet(block2, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
        builder.add_computational_memlet(block2, a, tasklet, "_in", {symbolic::symbol("i")}, desc_ptr);
        builder.add_computational_memlet(block2, tasklet, "_out", b, {symbolic::symbol("i")}, desc_ptr);
    }

    // Apply pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::BlockHoistingPass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    EXPECT_EQ(root.size(), 2);
    EXPECT_EQ(body.size(), 1);
}

TEST(BlockHoistingTest, For_InvariantView) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    types::Pointer opaque_ptr;
    builder.add_container("A", opaque_ptr, true);
    builder.add_container("a", opaque_ptr);
    builder.add_container("B", opaque_ptr, true);
    builder.add_container("i", sym_desc);

    auto& for_stmt = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::zero(),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );
    auto& body = for_stmt.root();

    // Loop invariant view
    auto& block1 = builder.add_block(body);
    {
        auto& a = builder.add_access(block1, "A");
        auto& a_ = builder.add_access(block1, "a");
        builder.add_reference_memlet(block1, a, a_, {symbolic::one()}, desc_ptr);
    }

    // Loop variant computation
    auto& block2 = builder.add_block(body);
    {
        auto& a = builder.add_access(block2, "a");
        auto& b = builder.add_access(block2, "B");
        auto& tasklet = builder.add_tasklet(block2, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
        builder.add_computational_memlet(block2, a, tasklet, "_in", {symbolic::symbol("i")}, desc_ptr);
        builder.add_computational_memlet(block2, tasklet, "_out", b, {symbolic::symbol("i")}, desc_ptr);
    }

    // Apply pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::BlockHoistingPass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    EXPECT_EQ(root.size(), 2);
    EXPECT_EQ(body.size(), 1);
}

TEST(BlockHoistingTest, For_InvariantView_MultipleViews) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    types::Pointer opaque_ptr;
    builder.add_container("A", opaque_ptr, true);
    builder.add_container("a1", opaque_ptr);
    builder.add_container("a2", opaque_ptr);
    builder.add_container("B", opaque_ptr, true);
    builder.add_container("i", sym_desc);

    auto& for_stmt = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::zero(),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );
    auto& body = for_stmt.root();

    // Loop invariant view
    auto& block1 = builder.add_block(body);
    {
        auto& a = builder.add_access(block1, "A");
        auto& a_ = builder.add_access(block1, "a1");
        builder.add_reference_memlet(block1, a, a_, {symbolic::zero()}, desc_ptr);
    }
    auto& block2 = builder.add_block(body);
    {
        auto& a = builder.add_access(block2, "A");
        auto& a_ = builder.add_access(block2, "a2");
        builder.add_reference_memlet(block2, a, a_, {symbolic::one()}, desc_ptr);
    }

    // Loop variant computation
    auto& block3 = builder.add_block(body);
    {
        auto& a1 = builder.add_access(block3, "a1");
        auto& a2 = builder.add_access(block3, "a2");
        auto& b = builder.add_access(block3, "B");
        auto& tasklet = builder.add_tasklet(block3, data_flow::TaskletCode::fp_add, {"_out"}, {"_in1", "_in2"});
        builder.add_computational_memlet(block3, a1, tasklet, "_in1", {symbolic::symbol("i")}, desc_ptr);
        builder.add_computational_memlet(block3, a2, tasklet, "_in2", {symbolic::symbol("i")}, desc_ptr);
        builder.add_computational_memlet(block3, tasklet, "_out", b, {symbolic::symbol("i")}, desc_ptr);
    }

    // Apply pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::BlockHoistingPass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    EXPECT_EQ(root.size(), 3);
    EXPECT_EQ(body.size(), 1);
}

TEST(BlockHoistingTest, IfElse_InvariantMove) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    types::Pointer opaque_ptr;
    builder.add_container("A", opaque_ptr, true);
    builder.add_container("a", opaque_ptr);
    builder.add_container("B", opaque_ptr, true);

    auto& if_else = builder.add_if_else(root);
    auto& case1 = builder.add_case(if_else, symbolic::Lt(symbolic::symbol("A"), symbolic::integer(5)));
    auto& case2 = builder.add_case(if_else, symbolic::Ge(symbolic::symbol("A"), symbolic::integer(5)));

    // Case 1: Branch invariant move
    auto& block1 = builder.add_block(case1);
    {
        auto& a = builder.add_access(block1, "A");
        auto& a_ = builder.add_access(block1, "a");
        builder.add_dereference_memlet(block1, a, a_, true, types::Pointer(static_cast<const types::IType&>(opaque_ptr)));
    }

    // Case 1: Branch variant computation
    auto& block2 = builder.add_block(case1);
    {
        auto& a = builder.add_access(block2, "a");
        auto& b = builder.add_access(block2, "B");
        auto& tasklet = builder.add_tasklet(block2, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
        builder.add_computational_memlet(block2, a, tasklet, "_in", {symbolic::integer(3)}, desc_ptr);
        builder.add_computational_memlet(block2, tasklet, "_out", b, {symbolic::integer(3)}, desc_ptr);
    }

    // Case 2: Branch invariant move
    auto& block3 = builder.add_block(case2);
    {
        auto& a = builder.add_access(block3, "A");
        auto& a_ = builder.add_access(block3, "a");
        builder.add_dereference_memlet(block3, a, a_, true, types::Pointer(static_cast<const types::IType&>(opaque_ptr)));
    }

    // Case 2: Branch variant computation
    auto& block4 = builder.add_block(case2);
    {
        auto& a = builder.add_access(block4, "a");
        auto& b = builder.add_access(block4, "B");
        auto& tasklet = builder.add_tasklet(block4, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
        builder.add_computational_memlet(block4, a, tasklet, "_in", {symbolic::integer(8)}, desc_ptr);
        builder.add_computational_memlet(block4, tasklet, "_out", b, {symbolic::integer(8)}, desc_ptr);
    }

    // Apply pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::BlockHoistingPass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    EXPECT_EQ(root.size(), 2);
    EXPECT_EQ(case1.size(), 1);
    EXPECT_EQ(case2.size(), 1);
}

TEST(BlockHoistingTest, IfElse_InvariantView) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    types::Pointer opaque_ptr;
    builder.add_container("A", opaque_ptr, true);
    builder.add_container("a", opaque_ptr);
    builder.add_container("B", opaque_ptr, true);

    auto& if_else = builder.add_if_else(root);
    auto& case1 = builder.add_case(if_else, symbolic::Lt(symbolic::symbol("A"), symbolic::integer(5)));
    auto& case2 = builder.add_case(if_else, symbolic::Ge(symbolic::symbol("A"), symbolic::integer(5)));

    // Case 1: Branch invariant view
    auto& block1 = builder.add_block(case1);
    {
        auto& a = builder.add_access(block1, "A");
        auto& a_ = builder.add_access(block1, "a");
        builder.add_reference_memlet(block1, a, a_, {symbolic::one()}, desc_ptr);
    }

    // Case 1: Branch variant computation
    auto& block2 = builder.add_block(case1);
    {
        auto& a = builder.add_access(block2, "a");
        auto& b = builder.add_access(block2, "B");
        auto& tasklet = builder.add_tasklet(block2, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
        builder.add_computational_memlet(block2, a, tasklet, "_in", {symbolic::integer(3)}, desc_ptr);
        builder.add_computational_memlet(block2, tasklet, "_out", b, {symbolic::integer(3)}, desc_ptr);
    }

    // Case 2: Branch invariant view
    auto& block3 = builder.add_block(case2);
    {
        auto& a = builder.add_access(block3, "A");
        auto& a_ = builder.add_access(block3, "a");
        builder.add_reference_memlet(block3, a, a_, {symbolic::one()}, desc_ptr);
    }

    // Case 2: Branch variant computation
    auto& block4 = builder.add_block(case2);
    {
        auto& a = builder.add_access(block4, "a");
        auto& b = builder.add_access(block4, "B");
        auto& tasklet = builder.add_tasklet(block4, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
        builder.add_computational_memlet(block4, a, tasklet, "_in", {symbolic::integer(8)}, desc_ptr);
        builder.add_computational_memlet(block4, tasklet, "_out", b, {symbolic::integer(8)}, desc_ptr);
    }

    // Apply pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::BlockHoistingPass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    EXPECT_EQ(root.size(), 2);
    EXPECT_EQ(case1.size(), 1);
    EXPECT_EQ(case2.size(), 1);
}

TEST(BlockHoistingTest, IfElse_InvariantView_MultipleViews) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    types::Pointer opaque_ptr;
    builder.add_container("A", opaque_ptr, true);
    builder.add_container("a1", opaque_ptr);
    builder.add_container("a2", opaque_ptr);
    builder.add_container("B", opaque_ptr, true);

    auto& if_else = builder.add_if_else(root);
    auto& case1 = builder.add_case(if_else, symbolic::Lt(symbolic::symbol("A"), symbolic::integer(5)));
    auto& case2 = builder.add_case(if_else, symbolic::Ge(symbolic::symbol("A"), symbolic::integer(5)));

    // Case 1: Branch invariant views
    auto& block1 = builder.add_block(case1);
    {
        auto& a = builder.add_access(block1, "A");
        auto& a_ = builder.add_access(block1, "a1");
        builder.add_reference_memlet(block1, a, a_, {symbolic::zero()}, desc_ptr);
    }
    auto& block2 = builder.add_block(case1);
    {
        auto& a = builder.add_access(block2, "A");
        auto& a_ = builder.add_access(block2, "a2");
        builder.add_reference_memlet(block2, a, a_, {symbolic::one()}, desc_ptr);
    }

    // Case 1: Branch variant computation
    auto& block3 = builder.add_block(case1);
    {
        auto& a1 = builder.add_access(block3, "a1");
        auto& a2 = builder.add_access(block3, "a2");
        auto& b = builder.add_access(block3, "B");
        auto& tasklet = builder.add_tasklet(block3, data_flow::TaskletCode::fp_add, {"_out"}, {"_in1", "_in2"});
        builder.add_computational_memlet(block3, a1, tasklet, "_in1", {symbolic::integer(3)}, desc_ptr);
        builder.add_computational_memlet(block3, a2, tasklet, "_in2", {symbolic::integer(3)}, desc_ptr);
        builder.add_computational_memlet(block3, tasklet, "_out", b, {symbolic::integer(3)}, desc_ptr);
    }

    // Case 2: Branch invariant views
    auto& block4 = builder.add_block(case2);
    {
        auto& a = builder.add_access(block4, "A");
        auto& a_ = builder.add_access(block4, "a1");
        builder.add_reference_memlet(block4, a, a_, {symbolic::zero()}, desc_ptr);
    }
    auto& block5 = builder.add_block(case2);
    {
        auto& a = builder.add_access(block5, "A");
        auto& a_ = builder.add_access(block5, "a2");
        builder.add_reference_memlet(block5, a, a_, {symbolic::one()}, desc_ptr);
    }

    // Case 2: Branch variant computation
    auto& block6 = builder.add_block(case2);
    {
        auto& a1 = builder.add_access(block6, "a1");
        auto& a2 = builder.add_access(block6, "a2");
        auto& b = builder.add_access(block6, "B");
        auto& tasklet = builder.add_tasklet(block6, data_flow::TaskletCode::fp_add, {"_out"}, {"_in1", "_in2"});
        builder.add_computational_memlet(block6, a1, tasklet, "_in1", {symbolic::integer(5)}, desc_ptr);
        builder.add_computational_memlet(block6, a2, tasklet, "_in2", {symbolic::integer(5)}, desc_ptr);
        builder.add_computational_memlet(block6, tasklet, "_out", b, {symbolic::integer(5)}, desc_ptr);
    }

    // Apply pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::BlockHoistingPass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    EXPECT_EQ(root.size(), 3);
    EXPECT_EQ(case1.size(), 1);
    EXPECT_EQ(case2.size(), 1);
}
