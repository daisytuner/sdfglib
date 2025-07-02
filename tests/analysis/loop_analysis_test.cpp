#include "sdfg/analysis/loop_analysis.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

using namespace sdfg;

TEST(LoopAnalysisTest, Monotonic) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("N", types::Scalar(types::PrimitiveType::Int32));

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto indvar = symbolic::symbol("i");
    // 3 * i + 2
    auto update = symbolic::add(symbolic::mul(symbolic::integer(3), indvar), symbolic::integer(2));
    auto condition = symbolic::Lt(indvar, symbolic::symbol("N"));
    auto init = symbolic::zero();
    auto& loop = builder.add_for(root, indvar, condition, init, update);

    analysis::AnalysisManager manager(sdfg);
    auto& assumptions_analysis = manager.get<analysis::AssumptionsAnalysis>();

    EXPECT_TRUE(analysis::LoopAnalysis::is_monotonic(&loop, assumptions_analysis));
    EXPECT_FALSE(analysis::LoopAnalysis::is_contiguous(&loop, assumptions_analysis));
}

TEST(LoopAnalysisTest, Contiguous) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("N", types::Scalar(types::PrimitiveType::Int32));

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto indvar = symbolic::symbol("i");
    // i + 1
    auto update = symbolic::add(indvar, symbolic::one());
    auto condition = symbolic::Lt(indvar, symbolic::symbol("N"));
    auto init = symbolic::zero();
    auto& loop = builder.add_for(root, indvar, condition, init, update);

    analysis::AnalysisManager manager(sdfg);
    auto& assumptions_analysis = manager.get<analysis::AssumptionsAnalysis>();

    EXPECT_TRUE(analysis::LoopAnalysis::is_monotonic(&loop, assumptions_analysis));
    EXPECT_TRUE(analysis::LoopAnalysis::is_contiguous(&loop, assumptions_analysis));
}

TEST(LoopAnalysisTest, CanonicalBound_Lt) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("N", types::Scalar(types::PrimitiveType::Int32));

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto indvar = symbolic::symbol("i");
    auto update = symbolic::add(indvar, symbolic::one());
    auto condition = symbolic::Lt(indvar, symbolic::symbol("N"));
    auto init = symbolic::zero();
    auto& loop = builder.add_for(root, indvar, condition, init, update);

    analysis::AnalysisManager manager(sdfg);
    auto& assumptions_analysis = manager.get<analysis::AssumptionsAnalysis>();

    auto bound = analysis::LoopAnalysis::canonical_bound(&loop, assumptions_analysis);
    EXPECT_TRUE(symbolic::eq(bound, symbolic::symbol("N")));
}

TEST(LoopAnalysisTest, CanonicalBound_Lt_And) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("N", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("M", types::Scalar(types::PrimitiveType::Int32));

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto indvar = symbolic::symbol("i");
    auto update = symbolic::add(indvar, symbolic::one());
    auto condition = symbolic::And(symbolic::Lt(indvar, symbolic::symbol("N")),
                                   symbolic::Lt(indvar, symbolic::symbol("M")));
    auto init = symbolic::zero();
    auto& loop = builder.add_for(root, indvar, condition, init, update);

    analysis::AnalysisManager manager(sdfg);
    auto& assumptions_analysis = manager.get<analysis::AssumptionsAnalysis>();

    auto bound = analysis::LoopAnalysis::canonical_bound(&loop, assumptions_analysis);
    EXPECT_TRUE(symbolic::eq(bound, symbolic::min(symbolic::symbol("N"), symbolic::symbol("M"))));
}

TEST(LoopAnalysisTest, CanonicalBound_Le_And) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("N", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("M", types::Scalar(types::PrimitiveType::Int32));

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto indvar = symbolic::symbol("i");
    auto update = symbolic::add(indvar, symbolic::one());
    auto condition = symbolic::And(symbolic::Le(indvar, symbolic::symbol("N")),
                                   symbolic::Le(indvar, symbolic::symbol("M")));
    auto init = symbolic::zero();
    auto& loop = builder.add_for(root, indvar, condition, init, update);

    analysis::AnalysisManager manager(sdfg);
    auto& assumptions_analysis = manager.get<analysis::AssumptionsAnalysis>();

    auto bound = analysis::LoopAnalysis::canonical_bound(&loop, assumptions_analysis);
    EXPECT_TRUE(
        symbolic::eq(bound, symbolic::min(symbolic::add(symbolic::symbol("N"), symbolic::one()),
                                          symbolic::add(symbolic::symbol("M"), symbolic::one()))));
}

TEST(LoopAnalysisTest, count_for_loops_0) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    analysis::AnalysisManager manager(sdfg);
    auto& loop_analysis = manager.get<analysis::LoopAnalysis>();

    EXPECT_EQ(loop_analysis.count_for_loops(), 0);
}

TEST(LoopAnalysisTest, count_for_loops_1) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto indvar = symbolic::symbol("i");
    auto update = symbolic::add(indvar, symbolic::one());
    auto condition = symbolic::And(symbolic::Le(indvar, symbolic::symbol("N")),
                                   symbolic::Le(indvar, symbolic::symbol("M")));
    auto init = symbolic::zero();
    auto& loop = builder.add_for(root, indvar, condition, init, update);

    analysis::AnalysisManager manager(sdfg);
    auto& loop_analysis = manager.get<analysis::LoopAnalysis>();

    EXPECT_EQ(loop_analysis.count_for_loops(), 1);
}

TEST(LoopAnalysisTest, count_for_loops_2) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto indvar1 = symbolic::symbol("i");
    auto update1 = symbolic::add(indvar1, symbolic::one());
    auto condition1 = symbolic::And(symbolic::Le(indvar1, symbolic::symbol("N")),
                                    symbolic::Le(indvar1, symbolic::symbol("M")));
    auto init1 = symbolic::zero();
    auto& loop1 = builder.add_for(root, indvar1, condition1, init1, update1);

    auto indvar2 = symbolic::symbol("j");
    auto update2 = symbolic::add(indvar2, symbolic::one());
    auto condition2 = symbolic::And(symbolic::Le(indvar2, symbolic::symbol("P")),
                                    symbolic::Le(indvar2, symbolic::symbol("Q")));
    auto init2 = symbolic::zero();
    auto& loop2 = builder.add_for(root, indvar2, condition2, init2, update2);

    analysis::AnalysisManager manager(sdfg);
    auto& loop_analysis = manager.get<analysis::LoopAnalysis>();

    EXPECT_EQ(loop_analysis.count_for_loops(), 2);
}

TEST(LoopAnalysisTest, count_for_loops_nested) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto indvar1 = symbolic::symbol("i");
    auto update1 = symbolic::add(indvar1, symbolic::one());
    auto condition1 = symbolic::And(symbolic::Le(indvar1, symbolic::symbol("N")),
                                    symbolic::Le(indvar1, symbolic::symbol("M")));
    auto init1 = symbolic::zero();
    auto& loop1 = builder.add_for(root, indvar1, condition1, init1, update1);

    auto indvar2 = symbolic::symbol("j");
    auto update2 = symbolic::add(indvar2, symbolic::one());
    auto condition2 = symbolic::And(symbolic::Le(indvar2, symbolic::symbol("P")),
                                    symbolic::Le(indvar2, symbolic::symbol("Q")));
    auto init2 = symbolic::zero();
    auto& loop2 = builder.add_for(loop1.root(), indvar2, condition2, init2, update2);

    analysis::AnalysisManager manager(sdfg);
    auto& loop_analysis = manager.get<analysis::LoopAnalysis>();

    EXPECT_EQ(loop_analysis.count_for_loops(), 2);
}

TEST(LoopAnalysisTest, count_for_loops_with_map) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto indvar1 = symbolic::symbol("i");
    auto update1 = symbolic::add(indvar1, symbolic::one());
    auto condition1 = symbolic::And(symbolic::Le(indvar1, symbolic::symbol("N")),
                                    symbolic::Le(indvar1, symbolic::symbol("M")));
    auto init1 = symbolic::zero();
    auto& loop1 = builder.add_for(root, indvar1, condition1, init1, update1);

    auto& map = builder.add_map(
        root, symbolic::symbol("i"), symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0), symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential);

    analysis::AnalysisManager manager(sdfg);
    auto& loop_analysis = manager.get<analysis::LoopAnalysis>();

    EXPECT_EQ(loop_analysis.count_for_loops(), 2);
}

TEST(LoopAnalysisTest, count_maps_0) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    analysis::AnalysisManager manager(sdfg);
    auto& loop_analysis = manager.get<analysis::LoopAnalysis>();

    EXPECT_EQ(loop_analysis.count_maps(), 0);
}

TEST(LoopAnalysisTest, count_maps_1) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto& map = builder.add_map(
        root, symbolic::symbol("i"), symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0), symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential);

    analysis::AnalysisManager manager(sdfg);
    auto& loop_analysis = manager.get<analysis::LoopAnalysis>();

    EXPECT_EQ(loop_analysis.count_maps(), 1);
}

TEST(LoopAnalysisTest, count_maps_2) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto& map1 = builder.add_map(
        root, symbolic::symbol("i"), symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0), symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential);

    auto& map2 = builder.add_map(
        root, symbolic::symbol("j"), symbolic::Lt(symbolic::symbol("j"), symbolic::integer(20)),
        symbolic::integer(0), symbolic::add(symbolic::symbol("j"), symbolic::integer(2)),
        structured_control_flow::ScheduleType_Sequential);

    analysis::AnalysisManager manager(sdfg);
    auto& loop_analysis = manager.get<analysis::LoopAnalysis>();

    EXPECT_EQ(loop_analysis.count_maps(), 2);
}

TEST(LoopAnalysisTest, count_maps_nested) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto& map1 = builder.add_map(
        root, symbolic::symbol("i"), symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0), symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential);

    auto& map2 = builder.add_map(
        map1.root(), symbolic::symbol("j"), symbolic::Lt(symbolic::symbol("j"), symbolic::integer(20)),
        symbolic::integer(0), symbolic::add(symbolic::symbol("j"), symbolic::integer(2)),
        structured_control_flow::ScheduleType_Sequential);

    analysis::AnalysisManager manager(sdfg);
    auto& loop_analysis = manager.get<analysis::LoopAnalysis>();

    EXPECT_EQ(loop_analysis.count_maps(), 2);
}

TEST(LoopAnalysisTest, count_maps_with_for) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto indvar1 = symbolic::symbol("i");
    auto update1 = symbolic::add(indvar1, symbolic::one());
    auto condition1 = symbolic::And(symbolic::Le(indvar1, symbolic::symbol("N")),
                                    symbolic::Le(indvar1, symbolic::symbol("M")));
    auto init1 = symbolic::zero();
    auto& loop1 = builder.add_for(root, indvar1, condition1, init1, update1);

    auto& map = builder.add_map(
        root, symbolic::symbol("i"), symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0), symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential);

    analysis::AnalysisManager manager(sdfg);
    auto& loop_analysis = manager.get<analysis::LoopAnalysis>();

    EXPECT_EQ(loop_analysis.count_maps(), 1);
}

TEST(LoopAnalysisTest, count_maps_with_schedule) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto& map = builder.add_map(
        root, symbolic::symbol("i"), symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0), symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential);

    analysis::AnalysisManager manager(sdfg);
    auto& loop_analysis = manager.get<analysis::LoopAnalysis>();

    EXPECT_EQ(loop_analysis.count_maps(structured_control_flow::ScheduleType_Sequential), 1);
}
