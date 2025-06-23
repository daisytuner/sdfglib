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
