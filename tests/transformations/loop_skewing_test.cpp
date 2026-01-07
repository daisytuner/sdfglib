#include "sdfg/transformations/loop_skewing.h"

#include <gtest/gtest.h>

using namespace sdfg;

TEST(LoopSkewingTest, Map_2D_Basic) {
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

    // Define outer loop (i)
    auto indvar_i = symbolic::symbol("i");
    auto& loop_i = builder.add_map(
        root,
        indvar_i,
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& body_i = loop_i.root();

    // Define inner loop (j)
    auto indvar_j = symbolic::symbol("j");
    auto& loop_j = builder.add_map(
        body_i,
        indvar_j,
        symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& body_j = loop_j.root();

    // Add computation
    auto& block = builder.add_block(body_j);
    auto& a_in = builder.add_access(block, "A");
    auto& one_node = builder.add_constant(block, "1.0", base_desc);
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(
        block, a_in, tasklet, "_in1", {symbolic::symbol("i"), symbolic::symbol("j")}, desc_2
    );
    builder.add_computational_memlet(block, one_node, tasklet, "_in2", {});
    builder.add_computational_memlet(
        block, tasklet, "_out", a_out, {symbolic::symbol("i"), symbolic::symbol("j")}, desc_2
    );

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::LoopSkewing transformation(loop_i, loop_j, 1);
    EXPECT_TRUE(transformation.can_be_applied(builder, analysis_manager));
    transformation.apply(builder, analysis_manager);

    auto& new_sdfg = builder.subject();
    EXPECT_EQ(new_sdfg.root().size(), 1);
    auto outer_loop = dynamic_cast<structured_control_flow::Map*>(&new_sdfg.root().at(0).first);
    EXPECT_TRUE(outer_loop != nullptr);
    auto inner_loop = dynamic_cast<structured_control_flow::Map*>(&outer_loop->root().at(0).first);
    EXPECT_TRUE(inner_loop != nullptr);

    // Verify that both loops still use their original induction variables
    EXPECT_EQ(outer_loop->indvar()->get_name(), "i");
    EXPECT_EQ(inner_loop->indvar()->get_name(), "j");

    // Verify structure
    EXPECT_EQ(outer_loop->root().size(), 1);
    EXPECT_EQ(inner_loop->root().size(), 1);
    EXPECT_EQ(&inner_loop->root().at(0).first, &block);
}

TEST(LoopSkewingTest, DependentLoops_ShouldFail) {
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
    auto& loop_i = builder.add_map(
        root,
        indvar_i,
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& body_i = loop_i.root();

    // Define inner loop that depends on outer loop
    auto indvar_j = symbolic::symbol("j");
    auto offset = symbolic::add(indvar_i, symbolic::integer(1));
    auto& loop_j = builder.add_map(
        body_i,
        indvar_j,
        symbolic::Lt(symbolic::symbol("j"), symbolic::sub(symbolic::symbol("M"), offset)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& body_j = loop_j.root();

    // Add computation
    auto& block = builder.add_block(body_j);
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(
        block, A_in, tasklet, "_in", {symbolic::symbol("i"), symbolic::add(symbolic::symbol("j"), offset)}, desc_2
    );
    builder.add_computational_memlet(
        block, tasklet, "_out", A_out, {symbolic::add(symbolic::symbol("j"), offset), symbolic::symbol("i")}, desc_2
    );

    // Analysis - should fail because inner loop depends on outer
    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::LoopSkewing transformation(loop_i, loop_j, 1);
    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));
}

TEST(LoopSkewingTest, OuterLoopHasMultipleChildren_ShouldFail) {
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
    auto indvar = symbolic::symbol("i");
    auto& loop = builder.add_map(
        root,
        indvar,
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& body = loop.root();
    
    // Add a block before the inner loop (this should make the transformation fail)
    auto& blocker = builder.add_block(body);

    // Define inner loop
    auto indvar_j = symbolic::symbol("j");
    auto& loop_j = builder.add_map(
        body,
        indvar_j,
        symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& body_j = loop_j.root();

    // Add computation
    auto& block = builder.add_block(body_j);
    auto& a_in = builder.add_access(block, "A");
    auto& i = builder.add_access(block, "i");
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(
        block, a_in, tasklet, "_in1", {symbolic::symbol("i"), symbolic::symbol("j")}, desc_2
    );
    builder.add_computational_memlet(block, i, tasklet, "_in2", {});
    builder.add_computational_memlet(
        block, tasklet, "_out", a_out, {symbolic::symbol("i"), symbolic::symbol("j")}, desc_2
    );

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::LoopSkewing transformation(loop, loop_j, 1);
    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));
}

TEST(LoopSkewingTest, ZeroSkewFactor_ShouldFail) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    // Define outer loop
    auto indvar_i = symbolic::symbol("i");
    auto& loop_i = builder.add_map(
        root,
        indvar_i,
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& body_i = loop_i.root();

    // Define inner loop
    auto indvar_j = symbolic::symbol("j");
    auto& loop_j = builder.add_map(
        body_i,
        indvar_j,
        symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    // Test with zero skew factor
    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::LoopSkewing transformation(loop_i, loop_j, 0);
    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));
}

TEST(LoopSkewingTest, NegativeSkewFactor) {
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
    auto& loop_i = builder.add_map(
        root,
        indvar_i,
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& body_i = loop_i.root();

    // Define inner loop
    auto indvar_j = symbolic::symbol("j");
    auto& loop_j = builder.add_map(
        body_i,
        indvar_j,
        symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& body_j = loop_j.root();

    // Add computation
    auto& block = builder.add_block(body_j);
    auto& a_in = builder.add_access(block, "A");
    auto& one_node = builder.add_constant(block, "1.0", base_desc);
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(
        block, a_in, tasklet, "_in1", {symbolic::symbol("i"), symbolic::symbol("j")}, desc_2
    );
    builder.add_computational_memlet(block, one_node, tasklet, "_in2", {});
    builder.add_computational_memlet(
        block, tasklet, "_out", a_out, {symbolic::symbol("i"), symbolic::symbol("j")}, desc_2
    );

    // Test with negative skew factor
    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::LoopSkewing transformation(loop_i, loop_j, -1);
    EXPECT_TRUE(transformation.can_be_applied(builder, analysis_manager));
    transformation.apply(builder, analysis_manager);

    auto& new_sdfg = builder.subject();
    EXPECT_EQ(new_sdfg.root().size(), 1);
}
