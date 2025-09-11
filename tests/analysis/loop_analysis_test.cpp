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
    auto condition =
        symbolic::And(symbolic::Lt(indvar, symbolic::symbol("N")), symbolic::Lt(indvar, symbolic::symbol("M")));
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
    auto condition =
        symbolic::And(symbolic::Le(indvar, symbolic::symbol("N")), symbolic::Le(indvar, symbolic::symbol("M")));
    auto init = symbolic::zero();
    auto& loop = builder.add_for(root, indvar, condition, init, update);

    analysis::AnalysisManager manager(sdfg);
    auto& assumptions_analysis = manager.get<analysis::AssumptionsAnalysis>();

    auto bound = analysis::LoopAnalysis::canonical_bound(&loop, assumptions_analysis);
    EXPECT_TRUE(symbolic::
                    eq(bound,
                       symbolic::
                           min(symbolic::add(symbolic::symbol("N"), symbolic::one()),
                               symbolic::add(symbolic::symbol("M"), symbolic::one()))));
}

TEST(LoopAnalysisTest, Children_nested3) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("j", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("k", types::Scalar(types::PrimitiveType::Int32));

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto indvar_i = symbolic::symbol("i");
    auto update_i = symbolic::add(indvar_i, symbolic::one());
    auto condition_i = symbolic::Lt(indvar_i, symbolic::symbol("N"));
    auto init_i = symbolic::zero();
    auto& loop_i = builder.add_for(root, indvar_i, condition_i, init_i, update_i);

    auto indvar_j = symbolic::symbol("j");
    auto update_j = symbolic::add(indvar_j, symbolic::one());
    auto condition_j = symbolic::Lt(indvar_j, symbolic::symbol("N"));
    auto init_j = symbolic::zero();
    auto& loop_j = builder.add_for(loop_i.root(), indvar_j, condition_j, init_j, update_j);

    auto indvar_k = symbolic::symbol("k");
    auto update_k = symbolic::add(indvar_k, symbolic::one());
    auto condition_k = symbolic::Lt(indvar_k, symbolic::symbol("N"));
    auto init_k = symbolic::zero();
    auto& loop_k = builder.add_for(loop_j.root(), indvar_k, condition_k, init_k, update_k);

    analysis::AnalysisManager manager(sdfg);
    auto& loop_analysis = manager.get<analysis::LoopAnalysis>();

    auto children_i = loop_analysis.children(&loop_i);
    EXPECT_EQ(children_i.size(), 1);
    EXPECT_EQ(children_i.at(0), &loop_j);

    auto children_j = loop_analysis.children(&loop_j);
    EXPECT_EQ(children_j.size(), 1);
    EXPECT_EQ(children_j.at(0), &loop_k);
    manager.invalidate_all();
}

TEST(LoopAnalysisTest, Children_nested2) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("j", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("k", types::Scalar(types::PrimitiveType::Int32));

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto indvar_i = symbolic::symbol("i");
    auto update_i = symbolic::add(indvar_i, symbolic::one());
    auto condition_i = symbolic::Lt(indvar_i, symbolic::symbol("N"));
    auto init_i = symbolic::zero();
    auto& loop_i = builder.add_for(root, indvar_i, condition_i, init_i, update_i);

    auto indvar_j = symbolic::symbol("j");
    auto update_j = symbolic::add(indvar_j, symbolic::one());
    auto condition_j = symbolic::Lt(indvar_j, symbolic::symbol("N"));
    auto init_j = symbolic::zero();
    auto& loop_j = builder.add_for(loop_i.root(), indvar_j, condition_j, init_j, update_j);

    auto indvar_k = symbolic::symbol("k");
    auto update_k = symbolic::add(indvar_k, symbolic::one());
    auto condition_k = symbolic::Lt(indvar_k, symbolic::symbol("N"));
    auto init_k = symbolic::zero();
    auto& loop_k = builder.add_for(loop_i.root(), indvar_k, condition_k, init_k, update_k);

    analysis::AnalysisManager manager(sdfg);
    auto& loop_analysis = manager.get<analysis::LoopAnalysis>();

    auto children_i = loop_analysis.children(&loop_i);
    EXPECT_EQ(children_i.size(), 2);
    bool found_j = false;
    bool found_k = false;
    for (auto node : children_i) {
        if (node == &loop_j) {
            found_j = true;
        } else if (node == &loop_k) {
            found_k = true;
        }
    }
    EXPECT_TRUE(found_j);
    EXPECT_TRUE(found_k);
    manager.invalidate_all();
}

TEST(LoopAnalysisTest, LoopTreePath_single) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("j", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("k", types::Scalar(types::PrimitiveType::Int32));

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto indvar_i = symbolic::symbol("i");
    auto update_i = symbolic::add(indvar_i, symbolic::one());
    auto condition_i = symbolic::Lt(indvar_i, symbolic::symbol("N"));
    auto init_i = symbolic::zero();
    auto& loop_i = builder.add_for(root, indvar_i, condition_i, init_i, update_i);

    auto indvar_j = symbolic::symbol("j");
    auto update_j = symbolic::add(indvar_j, symbolic::one());
    auto condition_j = symbolic::Lt(indvar_j, symbolic::symbol("N"));
    auto init_j = symbolic::zero();
    auto& loop_j = builder.add_for(loop_i.root(), indvar_j, condition_j, init_j, update_j);

    auto indvar_k = symbolic::symbol("k");
    auto update_k = symbolic::add(indvar_k, symbolic::one());
    auto condition_k = symbolic::Lt(indvar_k, symbolic::symbol("N"));
    auto init_k = symbolic::zero();
    auto& loop_k = builder.add_for(loop_j.root(), indvar_k, condition_k, init_k, update_k);

    analysis::AnalysisManager manager(sdfg);
    auto& loop_analysis = manager.get<analysis::LoopAnalysis>();

    auto path = loop_analysis.loop_tree_paths(&loop_i);
    EXPECT_EQ(path.size(), 1);
    EXPECT_EQ(path.back().size(), 3);
    EXPECT_EQ(path.back().at(0), &loop_i);
    EXPECT_EQ(path.back().at(1), &loop_j);
    EXPECT_EQ(path.back().at(2), &loop_k);
    manager.invalidate_all();
}

TEST(LoopAnalysisTest, LoopTreePath_split) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("j", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("k", types::Scalar(types::PrimitiveType::Int32));

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto indvar_i = symbolic::symbol("i");
    auto update_i = symbolic::add(indvar_i, symbolic::one());
    auto condition_i = symbolic::Lt(indvar_i, symbolic::symbol("N"));
    auto init_i = symbolic::zero();
    auto& loop_i = builder.add_for(root, indvar_i, condition_i, init_i, update_i);

    auto indvar_j = symbolic::symbol("j");
    auto update_j = symbolic::add(indvar_j, symbolic::one());
    auto condition_j = symbolic::Lt(indvar_j, symbolic::symbol("N"));
    auto init_j = symbolic::zero();
    auto& loop_j = builder.add_for(loop_i.root(), indvar_j, condition_j, init_j, update_j);

    auto indvar_k = symbolic::symbol("k");
    auto update_k = symbolic::add(indvar_k, symbolic::one());
    auto condition_k = symbolic::Lt(indvar_k, symbolic::symbol("N"));
    auto init_k = symbolic::zero();
    auto& loop_k = builder.add_for(loop_i.root(), indvar_k, condition_k, init_k, update_k);

    analysis::AnalysisManager manager(sdfg);
    auto& loop_analysis = manager.get<analysis::LoopAnalysis>();

    auto path = loop_analysis.loop_tree_paths(&loop_i);
    EXPECT_EQ(path.size(), 2);
    EXPECT_EQ(path.front().size(), 2);
    EXPECT_EQ(path.front().at(0), &loop_i);

    EXPECT_EQ(path.back().size(), 2);
    EXPECT_EQ(path.back().at(0), &loop_i);

    EXPECT_TRUE(
        path.front().at(1) == &loop_j && path.back().at(1) == &loop_k ||
        path.front().at(1) == &loop_k && path.back().at(1) == &loop_j
    );
    manager.invalidate_all();
}

TEST(LoopAnalysisTest, descendants_nested) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("j", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("k", types::Scalar(types::PrimitiveType::Int32));

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto indvar_i = symbolic::symbol("i");
    auto update_i = symbolic::add(indvar_i, symbolic::one());
    auto condition_i = symbolic::Lt(indvar_i, symbolic::symbol("N"));
    auto init_i = symbolic::zero();
    auto& loop_i = builder.add_for(root, indvar_i, condition_i, init_i, update_i);

    auto indvar_j = symbolic::symbol("j");
    auto update_j = symbolic::add(indvar_j, symbolic::one());
    auto condition_j = symbolic::Lt(indvar_j, symbolic::symbol("N"));
    auto init_j = symbolic::zero();
    auto& loop_j = builder.add_for(loop_i.root(), indvar_j, condition_j, init_j, update_j);

    auto indvar_k = symbolic::symbol("k");
    auto update_k = symbolic::add(indvar_k, symbolic::one());
    auto condition_k = symbolic::Lt(indvar_k, symbolic::symbol("N"));
    auto init_k = symbolic::zero();
    auto& loop_k = builder.add_for(loop_j.root(), indvar_k, condition_k, init_k, update_k);

    analysis::AnalysisManager manager(sdfg);
    auto& loop_analysis = manager.get<analysis::LoopAnalysis>();

    auto path = loop_analysis.descendants(&loop_i);
    EXPECT_EQ(path.size(), 2);
    bool found_j = false;
    bool found_k = false;
    for (auto& node : path) {
        if (node == &loop_j) {
            found_j = true;
        } else if (node == &loop_k) {
            found_k = true;
        }
    }
    EXPECT_TRUE(found_j);
    EXPECT_TRUE(found_k);
    manager.invalidate_all();
}

TEST(LoopAnalysisTest, descendants_concatenated) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("j", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("k", types::Scalar(types::PrimitiveType::Int32));

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto indvar_i = symbolic::symbol("i");
    auto update_i = symbolic::add(indvar_i, symbolic::one());
    auto condition_i = symbolic::Lt(indvar_i, symbolic::symbol("N"));
    auto init_i = symbolic::zero();
    auto& loop_i = builder.add_for(root, indvar_i, condition_i, init_i, update_i);

    auto indvar_j = symbolic::symbol("j");
    auto update_j = symbolic::add(indvar_j, symbolic::one());
    auto condition_j = symbolic::Lt(indvar_j, symbolic::symbol("N"));
    auto init_j = symbolic::zero();
    auto& loop_j = builder.add_for(loop_i.root(), indvar_j, condition_j, init_j, update_j);

    auto indvar_k = symbolic::symbol("k");
    auto update_k = symbolic::add(indvar_k, symbolic::one());
    auto condition_k = symbolic::Lt(indvar_k, symbolic::symbol("N"));
    auto init_k = symbolic::zero();
    auto& loop_k = builder.add_for(loop_i.root(), indvar_k, condition_k, init_k, update_k);

    analysis::AnalysisManager manager(sdfg);
    auto& loop_analysis = manager.get<analysis::LoopAnalysis>();

    auto path = loop_analysis.descendants(&loop_i);
    EXPECT_EQ(path.size(), 2);
    bool found_j = false;
    bool found_k = false;
    for (auto node : path) {
        if (node == &loop_j) {
            found_j = true;
        } else if (node == &loop_k) {
            found_k = true;
        }
    }
    EXPECT_TRUE(found_j);
    EXPECT_TRUE(found_k);
    manager.invalidate_all();
}
