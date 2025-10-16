#include "sdfg/analysis/flop_analysis.h"

#include <gtest/gtest.h>
#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/intrinsic.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/element.h"
#include "sdfg/function.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"

using namespace sdfg;

TEST(FlopAnalysis, Tasklet) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc(types::PrimitiveType::Float);
    builder.add_container("a", desc);
    builder.add_container("b", desc);
    builder.add_container("c", desc);

    // Add block with tasklet
    auto& block = builder.add_block(root);
    auto& a = builder.add_access(block, "a");
    auto& b = builder.add_access(block, "b");
    auto& c = builder.add_access(block, "c");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, {"_out"}, {"_in1", "_in2"});
    builder.add_computational_memlet(block, a, tasklet, "_in1", {});
    builder.add_computational_memlet(block, b, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", c, {});

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::FlopAnalysis>();

    // Check
    symbolic::Expression flop;
    ASSERT_TRUE(analysis.contains(&block));
    flop = analysis.get(&block);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::one()));
    ASSERT_TRUE(analysis.contains(&root));
    flop = analysis.get(&root);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::one()));
}

TEST(FlopAnalysis, MultipleTasklets) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc(types::PrimitiveType::Float);
    builder.add_container("a", desc);
    builder.add_container("b", desc);
    builder.add_container("c", desc);
    builder.add_container("d", desc);
    builder.add_container("e", desc);
    builder.add_container("f", desc);

    // Add block with two tasklets
    auto& block = builder.add_block(root);
    auto& a = builder.add_access(block, "a");
    auto& b = builder.add_access(block, "b");
    auto& c = builder.add_access(block, "c");
    auto& d = builder.add_access(block, "d");
    auto& e = builder.add_access(block, "e");
    auto& f = builder.add_access(block, "f");
    auto& tasklet1 = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, {"_out"}, {"_in1", "_in2"});
    builder.add_computational_memlet(block, a, tasklet1, "_in1", {});
    builder.add_computational_memlet(block, b, tasklet1, "_in2", {});
    builder.add_computational_memlet(block, tasklet1, "_out", c, {});
    auto& tasklet2 = builder.add_tasklet(block, data_flow::TaskletCode::fp_fma, {"_out"}, {"_in1", "_in2", "_in3"});
    builder.add_computational_memlet(block, c, tasklet2, "_in1", {});
    builder.add_computational_memlet(block, d, tasklet2, "_in2", {});
    builder.add_computational_memlet(block, e, tasklet2, "_in3", {});
    builder.add_computational_memlet(block, tasklet2, "_out", f, {});

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::FlopAnalysis>();

    // Check
    symbolic::Expression flop;
    ASSERT_TRUE(analysis.contains(&block));
    flop = analysis.get(&block);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::integer(3)));
    ASSERT_TRUE(analysis.contains(&root));
    flop = analysis.get(&root);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::integer(3)));
}

TEST(FlopAnalysis, Loop) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc(types::PrimitiveType::Float);
    builder.add_container("a", desc);
    builder.add_container("b", desc);
    builder.add_container("c", desc);

    // Add loop
    auto i = symbolic::symbol("i");
    auto n = symbolic::symbol("n");
    auto& loop = builder.add_for(root, i, symbolic::Lt(i, n), symbolic::zero(), symbolic::add(i, symbolic::one()));

    // Add block with tasklet
    auto& block = builder.add_block(loop.root());
    auto& a = builder.add_access(block, "a");
    auto& b = builder.add_access(block, "b");
    auto& c = builder.add_access(block, "c");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, {"_out"}, {"_in1", "_in2"});
    builder.add_computational_memlet(block, a, tasklet, "_in1", {});
    builder.add_computational_memlet(block, b, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", c, {});

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::FlopAnalysis>();

    // Check
    symbolic::Expression flop;
    ASSERT_TRUE(analysis.contains(&block));
    flop = analysis.get(&block);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::one()));
    ASSERT_TRUE(analysis.contains(&loop.root()));
    flop = analysis.get(&loop.root());
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::one()));
    ASSERT_TRUE(analysis.contains(&loop));
    flop = analysis.get(&loop);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::symbol("n")));
    ASSERT_TRUE(analysis.contains(&root));
    flop = analysis.get(&root);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::symbol("n")));
}

TEST(FlopAnalysis, LoopNest) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc(types::PrimitiveType::Float);
    builder.add_container("a", desc);
    builder.add_container("b", desc);
    builder.add_container("c", desc);
    builder.add_container("d", desc);

    // Add first loop
    auto i = symbolic::symbol("i");
    auto n = symbolic::symbol("n");
    auto& loop1 = builder.add_for(root, i, symbolic::Lt(i, n), symbolic::zero(), symbolic::add(i, symbolic::one()));

    // Add second loop, bound: i <= N && i < M, update: j = j + 2
    auto j = symbolic::symbol("j");
    auto bound2 = symbolic::And(symbolic::Le(j, symbolic::symbol("m")), symbolic::Lt(j, symbolic::symbol("k")));
    auto& loop2 = builder.add_for(loop1.root(), j, bound2, symbolic::one(), symbolic::add(j, symbolic::integer(2)));

    // Add block with tasklet
    auto& block = builder.add_block(loop2.root());
    auto& a = builder.add_access(block, "a");
    auto& b = builder.add_access(block, "b");
    auto& c = builder.add_access(block, "c");
    auto& d = builder.add_access(block, "d");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_fma, {"_out"}, {"_in1", "_in2", "_in3"});
    builder.add_computational_memlet(block, a, tasklet, "_in1", {});
    builder.add_computational_memlet(block, b, tasklet, "_in2", {});
    builder.add_computational_memlet(block, c, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", d, {});

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::FlopAnalysis>();

    // Check
    symbolic::Expression flop;
    ASSERT_TRUE(analysis.contains(&block));
    flop = analysis.get(&block);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::integer(2)));
    ASSERT_TRUE(analysis.contains(&loop2.root()));
    flop = analysis.get(&loop2.root());
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::integer(2)));
    ASSERT_TRUE(analysis.contains(&loop2));
    flop = analysis.get(&loop2);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::parse("idiv(min(m + 1, k) - 1, 2) * 2")));
    ASSERT_TRUE(analysis.contains(&loop1.root()));
    flop = analysis.get(&loop1.root());
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::parse("idiv(min(m + 1, k) - 1, 2) * 2")));
    ASSERT_TRUE(analysis.contains(&loop1));
    flop = analysis.get(&loop1);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::parse("n * idiv(min(m + 1, k) - 1, 2) * 2")));
    ASSERT_TRUE(analysis.contains(&root));
    flop = analysis.get(&root);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::parse("n * idiv(min(m + 1, k) - 1, 2) * 2")));
}

TEST(FlopAnalysis, Intrinsic) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc(types::PrimitiveType::Float);
    builder.add_container("a", desc);
    builder.add_container("b", desc);

    // Add block with library node
    auto& block = builder.add_block(root);
    auto& a = builder.add_access(block, "a");
    auto& b = builder.add_access(block, "b");
    auto& libnode = builder.add_library_node<math::IntrinsicNode>(block, DebugInfo(), "sin", 1);
    builder.add_computational_memlet(block, a, libnode, "_in1", {}, desc);
    builder.add_computational_memlet(block, libnode, "_out", b, {}, desc);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::FlopAnalysis>();

    // Check
    symbolic::Expression flop;
    ASSERT_TRUE(analysis.contains(&block));
    flop = analysis.get(&block);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::one()));
    ASSERT_TRUE(analysis.contains(&root));
    flop = analysis.get(&root);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::one()));
}

TEST(FlopAnalysis, IfElse) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc(types::PrimitiveType::Float);
    builder.add_container("a", desc);
    builder.add_container("b", desc);
    builder.add_container("c", desc);
    builder.add_container("d", desc);

    // Add IfElse
    auto& if_else = builder.add_if_else(root);
    auto& case1 = builder.add_case(if_else, symbolic::Eq(symbolic::symbol("a"), symbolic::zero()));
    auto& case2 = builder.add_case(if_else, symbolic::Ne(symbolic::symbol("a"), symbolic::zero()));

    // Add first block with tasklet
    auto& block1 = builder.add_block(case1);
    {
        auto& b = builder.add_access(block1, "b");
        auto& c = builder.add_access(block1, "c");
        auto& d = builder.add_access(block1, "d");
        auto& tasklet = builder.add_tasklet(block1, data_flow::TaskletCode::fp_add, {"_out"}, {"_in1", "_in2"});
        builder.add_computational_memlet(block1, b, tasklet, "_in1", {});
        builder.add_computational_memlet(block1, c, tasklet, "_in2", {});
        builder.add_computational_memlet(block1, tasklet, "_out", d, {});
    }

    // Add second block with tasklet
    auto& block2 = builder.add_block(case2);
    {
        auto& a = builder.add_access(block2, "a");
        auto& b = builder.add_access(block2, "b");
        auto& c = builder.add_access(block2, "c");
        auto& d = builder.add_access(block2, "d");
        auto& tasklet = builder.add_tasklet(block2, data_flow::TaskletCode::fp_fma, {"_out"}, {"_in1", "_in2", "_in3"});
        builder.add_computational_memlet(block2, a, tasklet, "_in1", {});
        builder.add_computational_memlet(block2, b, tasklet, "_in2", {});
        builder.add_computational_memlet(block2, c, tasklet, "_in3", {});
        builder.add_computational_memlet(block2, tasklet, "_out", d, {});
    }

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::FlopAnalysis>();

    // Check
    symbolic::Expression flop;
    ASSERT_TRUE(analysis.contains(&block1));
    flop = analysis.get(&block1);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::one()));
    ASSERT_TRUE(analysis.contains(&block2));
    flop = analysis.get(&block2);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::integer(2)));
    ASSERT_TRUE(analysis.contains(&if_else));
    flop = analysis.get(&if_else);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::max(symbolic::one(), symbolic::integer(2))));
    ASSERT_TRUE(analysis.contains(&root));
    flop = analysis.get(&root);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::max(symbolic::one(), symbolic::integer(2))));
}

TEST(FlopAnalysis, WhileLoop) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc(types::PrimitiveType::Float);
    builder.add_container("a", desc);
    builder.add_container("b", desc);
    builder.add_container("c", desc);

    // Add loop
    auto& loop = builder.add_while(root);

    // Add block with tasklet
    auto& block = builder.add_block(loop.root());
    auto& a = builder.add_access(block, "a");
    auto& b = builder.add_access(block, "b");
    auto& c = builder.add_access(block, "c");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, {"_out"}, {"_in1", "_in2"});
    builder.add_computational_memlet(block, a, tasklet, "_in1", {});
    builder.add_computational_memlet(block, b, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", c, {});

    // Add break
    auto& break_node = builder.add_break(loop.root());

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::FlopAnalysis>();

    // Check
    symbolic::Expression flop;
    ASSERT_TRUE(analysis.contains(&block));
    flop = analysis.get(&block);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::one()));
    ASSERT_TRUE(analysis.contains(&break_node));
    flop = analysis.get(&break_node);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::zero()));
    ASSERT_TRUE(analysis.contains(&loop.root()));
    flop = analysis.get(&loop.root());
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::one()));
    ASSERT_TRUE(analysis.contains(&loop));
    flop = analysis.get(&loop);
    ASSERT_TRUE(flop.is_null());
    ASSERT_TRUE(analysis.contains(&root));
    flop = analysis.get(&root);
    ASSERT_TRUE(flop.is_null());
}

TEST(FlopAnalysis, LoopIndvarDependency) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc(types::PrimitiveType::Float);
    builder.add_container("a", desc);
    builder.add_container("b", desc);
    builder.add_container("c", desc);

    // Add first loop
    auto i = symbolic::symbol("i");
    auto n = symbolic::symbol("n");
    auto& loop1 = builder.add_for(root, i, symbolic::Lt(i, n), symbolic::zero(), symbolic::add(i, symbolic::one()));

    // Add second loop
    auto j = symbolic::symbol("j");
    auto m = symbolic::symbol("m");
    auto& loop2 = builder.add_for(loop1.root(), j, symbolic::Lt(j, m), i, symbolic::add(j, symbolic::one()));

    // Add block with tasklet
    auto& block = builder.add_block(loop2.root());
    auto& a = builder.add_access(block, "a");
    auto& b = builder.add_access(block, "b");
    auto& c = builder.add_access(block, "c");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, {"_out"}, {"_in1", "_in2"});
    builder.add_computational_memlet(block, a, tasklet, "_in1", {});
    builder.add_computational_memlet(block, b, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", c, {});

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::FlopAnalysis>();

    // Check
    symbolic::Expression flop;
    ASSERT_TRUE(analysis.contains(&block));
    flop = analysis.get(&block);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::one()));
    ASSERT_TRUE(analysis.contains(&loop2.root()));
    flop = analysis.get(&loop2.root());
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::one()));
    ASSERT_TRUE(analysis.contains(&loop2));
    flop = analysis.get(&loop2);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::parse("m - idiv(n - 1, 2)")));
    ASSERT_TRUE(analysis.contains(&loop1.root()));
    flop = analysis.get(&loop1.root());
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::parse("m - idiv(n - 1, 2)")));
    ASSERT_TRUE(analysis.contains(&loop1));
    flop = analysis.get(&loop1);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::parse("n * (m - idiv(n - 1, 2))")));
    ASSERT_TRUE(analysis.contains(&root));
    flop = analysis.get(&root);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::parse("n * (m - idiv(n - 1, 2))")));
}
