#include "sdfg/transformations/loop_interchange.h"

#include <gtest/gtest.h>

using namespace sdfg;

TEST(LoopInterchangeTest, Map_2D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU, DebugTable());

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

    // Define loop 1
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");

    auto& loop = builder.add_map(
        root,
        indvar,
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential
    );
    auto& body = loop.root();

    // Define loop 2
    auto bound_2 = symbolic::symbol("M");
    auto indvar_2 = symbolic::symbol("j");

    auto& loop_2 = builder.add_map(
        body,
        indvar_2,
        symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential
    );
    auto& body_2 = loop_2.root();

    // Add computation
    auto& block = builder.add_block(body_2);
    auto& a_in = builder.add_access(block, "A");
    auto& i = builder.add_access(block, "i");
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::add, "_out", {"_in1", "_in2"});
    builder
        .add_computational_memlet(block, a_in, tasklet, "_in1", {symbolic::symbol("i"), symbolic::symbol("j")}, desc_2);
    builder.add_computational_memlet(block, i, tasklet, "_in2", {});
    builder
        .add_computational_memlet(block, tasklet, "_out", a_out, {symbolic::symbol("i"), symbolic::symbol("j")}, desc_2);

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::LoopInterchange transformation(loop, loop_2);
    EXPECT_TRUE(transformation.can_be_applied(builder, analysis_manager));
    transformation.apply(builder, analysis_manager);

    auto& new_sdfg = builder.subject();
    EXPECT_EQ(new_sdfg.root().size(), 1);
    auto outer_loop = dynamic_cast<structured_control_flow::Map*>(&new_sdfg.root().at(0).first);
    EXPECT_TRUE(outer_loop != nullptr);
    auto inner_loop = dynamic_cast<structured_control_flow::Map*>(&outer_loop->root().at(0).first);
    EXPECT_TRUE(inner_loop != nullptr);

    EXPECT_EQ(outer_loop->indvar()->get_name(), "j");
    EXPECT_EQ(inner_loop->indvar()->get_name(), "i");

    EXPECT_EQ(outer_loop->root().size(), 1);
    EXPECT_EQ(inner_loop->root().size(), 1);
    EXPECT_EQ(&inner_loop->root().at(0).first, &block);
}

TEST(LoopInterchangeTest, DependentLoops) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU, DebugTable());

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

    // Define loop
    auto indvar1 = symbolic::symbol("i");
    auto& loop1 = builder.add_map(
        root,
        indvar1,
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential
    );
    auto& body1 = loop1.root();

    auto indvar2 = symbolic::symbol("j");
    auto offset2 = symbolic::add(indvar1, symbolic::integer(1));
    auto& loop2 = builder.add_map(
        body1,
        indvar2,
        symbolic::Lt(symbolic::symbol("j"), symbolic::sub(symbolic::symbol("M"), offset2)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential
    );
    auto& body2 = loop2.root();

    // Add computation
    auto& block = builder.add_block(body2);
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(
        block, A_in, tasklet, "_in", {symbolic::symbol("i"), symbolic::add(symbolic::symbol("j"), offset2)}, desc_2
    );
    builder.add_computational_memlet(
        block, tasklet, "_out", A_out, {symbolic::add(symbolic::symbol("j"), offset2), symbolic::symbol("i")}, desc_2
    );

    // Analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::LoopInterchange transformation(loop1, loop2);
    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));
}

TEST(LoopInterchangeTest, OuterLoopHasOuterBlocks) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU, DebugTable());

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

    // Define loop 1
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");

    auto& loop = builder.add_map(
        root,
        indvar,
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential
    );
    auto& body = loop.root();
    auto& blocker = builder.add_block(body);

    // Define loop 2
    auto bound_2 = symbolic::symbol("M");
    auto indvar_2 = symbolic::symbol("j");

    auto& loop_2 = builder.add_map(
        body,
        indvar_2,
        symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential
    );
    auto& body_2 = loop_2.root();

    // Add computation
    auto& block = builder.add_block(body_2);
    auto& a_in = builder.add_access(block, "A");
    auto& i = builder.add_access(block, "i");
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::add, "_out", {"_in1", "_in2"});
    builder
        .add_computational_memlet(block, a_in, tasklet, "_in1", {symbolic::symbol("i"), symbolic::symbol("j")}, desc_2);
    builder.add_computational_memlet(block, i, tasklet, "_in2", {});
    builder
        .add_computational_memlet(block, tasklet, "_out", a_out, {symbolic::symbol("i"), symbolic::symbol("j")}, desc_2);

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::LoopInterchange transformation(loop, loop_2);
    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));
}
