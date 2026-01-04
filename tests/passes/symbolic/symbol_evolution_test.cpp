#include "sdfg/passes/symbolic/symbol_evolution.h"

#include <gtest/gtest.h>

#include <ostream>

#include "sdfg/builder/structured_sdfg_builder.h"

using namespace sdfg;

// Test Pattern 1: Constant - Positive Case
// Symbol is assigned a constant value and never changes
TEST(SymbolEvolutionTest, Pattern1_Constant_Positive) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    builder.add_container("c", desc);
    auto i = symbolic::symbol("i");
    auto c = symbolic::symbol("c");

    auto& root = builder.subject().root();

    // c = 42
    auto& init_block = builder.add_block(root);
    init_block.dataflow();

    builder.add_block_after(root, init_block, {{c, symbolic::integer(42)}}, init_block.debug_info());

    // for (i = 0; i < 10; i++) { c = 42; }
    auto& loop =
        builder
            .add_for(root, i, symbolic::Lt(i, symbolic::integer(10)), symbolic::zero(), symbolic::add(i, symbolic::one()));

    auto& loop_block = builder.add_block(loop.root());
    loop_block.dataflow();

    builder.add_block_after(loop.root(), loop_block, {{c, symbolic::integer(42)}}, loop_block.debug_info());

    auto& after_block = builder.add_block(loop.root());
    after_block.dataflow();

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::SymbolEvolution pass;
    bool applied = pass.run(builder, analysis_manager);

    EXPECT_TRUE(applied);

    // Verify: c should be redefined at the beginning of the loop
    EXPECT_EQ(loop.root().size(), 3);
}

// Test Pattern 1: Constant - Negative Case
// Symbol changes value, so constant pattern doesn't apply
TEST(SymbolEvolutionTest, Pattern1_Constant_Negative) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    builder.add_container("c", desc);
    auto i = symbolic::symbol("i");
    auto c = symbolic::symbol("c");

    auto& root = builder.subject().root();

    // c = 42
    auto& init_block = builder.add_block(root);
    init_block.dataflow();

    builder.add_block_after(root, init_block, {{c, symbolic::integer(42)}}, init_block.debug_info());

    // for (i = 0; i < 10; i++) { c = 100; } // Different value
    auto& loop =
        builder
            .add_for(root, i, symbolic::Lt(i, symbolic::integer(10)), symbolic::zero(), symbolic::add(i, symbolic::one()));

    auto& loop_block = builder.add_block(loop.root());
    loop_block.dataflow();

    builder.add_block_after(loop.root(), loop_block, {{c, symbolic::integer(100)}}, loop_block.debug_info());

    auto& after_block = builder.add_block(loop.root());
    after_block.dataflow();

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::SymbolEvolution pass;
    bool applied = pass.run(builder, analysis_manager);

    EXPECT_FALSE(applied);
}

// Test Pattern 2: Loop Alias - Positive Case
// Symbol mimics the loop induction variable
TEST(SymbolEvolutionTest, Pattern2_LoopAlias_Positive) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    builder.add_container("j", desc);
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");

    auto& root = builder.subject().root();

    // j = 0
    auto& init_block = builder.add_block(root);
    init_block.dataflow();

    builder.add_block_after(root, init_block, {{j, symbolic::zero()}}, init_block.debug_info());

    // for (i = 0; i < 10; i++) { j = i + 1; }
    auto& loop =
        builder
            .add_for(root, i, symbolic::Lt(i, symbolic::integer(10)), symbolic::zero(), symbolic::add(i, symbolic::one()));

    auto& loop_block = builder.add_block(loop.root());
    loop_block.dataflow();

    builder.add_block_after(loop.root(), loop_block, {{j, symbolic::add(i, symbolic::one())}}, loop_block.debug_info());

    auto& after_block = builder.add_block(loop.root());
    after_block.dataflow();

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::SymbolEvolution pass;
    bool applied = pass.run(builder, analysis_manager);

    EXPECT_TRUE(applied);
}

// Test Pattern 2: Loop Alias - Negative Case
// Symbol update doesn't match loop update
TEST(SymbolEvolutionTest, Pattern2_LoopAlias_Negative) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    builder.add_container("j", desc);
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");

    auto& root = builder.subject().root();

    // j = 0
    auto& init_block = builder.add_block(root);
    init_block.dataflow();

    builder.add_block_after(root, init_block, {{j, symbolic::zero()}}, init_block.debug_info());

    // for (i = 0; i < 10; i++) { j = i + 2; } // Different update
    auto& loop =
        builder
            .add_for(root, i, symbolic::Lt(i, symbolic::integer(10)), symbolic::zero(), symbolic::add(i, symbolic::one()));

    auto& loop_block = builder.add_block(loop.root());
    loop_block.dataflow();

    builder
        .add_block_after(loop.root(), loop_block, {{j, symbolic::add(i, symbolic::integer(2))}}, loop_block.debug_info());

    auto& after_block = builder.add_block(loop.root());
    after_block.dataflow();

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::SymbolEvolution pass;
    bool applied = pass.run(builder, analysis_manager);

    EXPECT_FALSE(applied);
}

// Test Pattern 3: Affine Update - Positive Case
// Symbol increases by a constant each iteration
TEST(SymbolEvolutionTest, Pattern3_AffineUpdate_Positive) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    builder.add_container("sum", desc);
    auto i = symbolic::symbol("i");
    auto sum = symbolic::symbol("sum");

    auto& root = builder.subject().root();

    // sum = 0
    auto& init_block = builder.add_block(root);
    init_block.dataflow();

    builder.add_block_after(root, init_block, {{sum, symbolic::zero()}}, init_block.debug_info());

    // for (i = 0; i < 10; i++) { sum = sum + 5; }
    auto& loop =
        builder
            .add_for(root, i, symbolic::Lt(i, symbolic::integer(10)), symbolic::zero(), symbolic::add(i, symbolic::one()));

    auto& loop_block = builder.add_block(loop.root());
    loop_block.dataflow();

    builder.add_block_after(
        loop.root(), loop_block, {{sum, symbolic::add(sum, symbolic::integer(5))}}, loop_block.debug_info()
    );

    auto& after_block = builder.add_block(loop.root());
    after_block.dataflow();

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::SymbolEvolution pass;
    bool applied = pass.run(builder, analysis_manager);

    EXPECT_TRUE(applied);

    // Verify: sum should be redefined as 5 * i
    EXPECT_EQ(loop.root().size(), 3);
}

// Test Pattern 3: Affine Update - Negative Case
// Symbol has multiplicative update (not affine addition)
TEST(SymbolEvolutionTest, Pattern3_AffineUpdate_Negative_Multiplicative) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    builder.add_container("prod", desc);
    auto i = symbolic::symbol("i");
    auto prod = symbolic::symbol("prod");

    auto& root = builder.subject().root();

    // prod = 1
    auto& init_block = builder.add_block(root);
    init_block.dataflow();

    builder.add_block_after(root, init_block, {{prod, symbolic::one()}}, init_block.debug_info());

    // for (i = 0; i < 10; i++) { prod = prod * 2; }
    auto& loop =
        builder
            .add_for(root, i, symbolic::Lt(i, symbolic::integer(10)), symbolic::zero(), symbolic::add(i, symbolic::one()));

    auto& loop_block = builder.add_block(loop.root());
    loop_block.dataflow();

    builder.add_block_after(
        loop.root(), loop_block, {{prod, symbolic::mul(prod, symbolic::integer(2))}}, loop_block.debug_info()
    );

    auto& after_block = builder.add_block(loop.root());
    after_block.dataflow();

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::SymbolEvolution pass;
    bool applied = pass.run(builder, analysis_manager);

    EXPECT_FALSE(applied);
}

// Test Pattern 4: Loop-dependent function - Positive Case
// Symbol is a function of the previous iteration's induction variable
TEST(SymbolEvolutionTest, Pattern4_LoopDependentFunction_Positive) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    builder.add_container("offset", desc);
    auto i = symbolic::symbol("i");
    auto offset = symbolic::symbol("offset");

    auto& root = builder.subject().root();

    // offset = -1 (which is 0 - 1, matching initial condition)
    auto& init_block = builder.add_block(root);
    init_block.dataflow();

    builder.add_block_after(root, init_block, {{offset, symbolic::integer(-1)}}, init_block.debug_info());

    // for (i = 0; i < 10; i++) { offset = i; }
    auto& loop =
        builder
            .add_for(root, i, symbolic::Lt(i, symbolic::integer(10)), symbolic::zero(), symbolic::add(i, symbolic::one()));

    auto& loop_block = builder.add_block(loop.root());
    loop_block.dataflow();

    builder.add_block_after(loop.root(), loop_block, {{offset, i}}, loop_block.debug_info());

    auto& after_block = builder.add_block(loop.root());
    after_block.dataflow();

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::SymbolEvolution pass;
    bool applied = pass.run(builder, analysis_manager);

    EXPECT_TRUE(applied);
}

// Test Pattern 4: Loop-dependent function - Negative Case
// Initial condition doesn't match
TEST(SymbolEvolutionTest, Pattern4_LoopDependentFunction_Negative) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    builder.add_container("offset", desc);
    auto i = symbolic::symbol("i");
    auto offset = symbolic::symbol("offset");

    auto& root = builder.subject().root();

    // offset = 10 (doesn't match initial condition)
    auto& init_block = builder.add_block(root);
    init_block.dataflow();

    builder.add_block_after(root, init_block, {{offset, symbolic::integer(10)}}, init_block.debug_info());

    // for (i = 0; i < 10; i++) { offset = i; }
    auto& loop =
        builder
            .add_for(root, i, symbolic::Lt(i, symbolic::integer(10)), symbolic::zero(), symbolic::add(i, symbolic::one()));

    auto& loop_block = builder.add_block(loop.root());
    loop_block.dataflow();

    builder.add_block_after(loop.root(), loop_block, {{offset, i}}, loop_block.debug_info());

    auto& after_block = builder.add_block(loop.root());
    after_block.dataflow();

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::SymbolEvolution pass;
    bool applied = pass.run(builder, analysis_manager);

    EXPECT_FALSE(applied);
}

// Test: Symbol used after update - Negative Case
// Symbol is read after being updated in the loop, so it can't be optimized
TEST(SymbolEvolutionTest, UsedAfterUpdate_Negative) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    builder.add_container("sum", desc);
    builder.add_container("temp", desc);
    auto i = symbolic::symbol("i");
    auto sum = symbolic::symbol("sum");
    auto temp = symbolic::symbol("temp");

    auto& root = builder.subject().root();

    // sum = 0
    auto& init_block = builder.add_block(root);
    init_block.dataflow();

    builder.add_block_after(root, init_block, {{sum, symbolic::zero()}}, init_block.debug_info());

    // for (i = 0; i < 10; i++) { sum = sum + 5; temp = sum; }
    auto& loop =
        builder
            .add_for(root, i, symbolic::Lt(i, symbolic::integer(10)), symbolic::zero(), symbolic::add(i, symbolic::one()));

    auto& loop_block1 = builder.add_block(loop.root());
    loop_block1.dataflow();

    builder.add_block_after(
        loop.root(), loop_block1, {{sum, symbolic::add(sum, symbolic::integer(5))}}, loop_block1.debug_info()
    );

    auto& loop_block2 = builder.add_block(loop.root());
    loop_block2.dataflow();

    builder.add_block_after(loop.root(), loop_block2, {{temp, sum}}, loop_block2.debug_info());

    auto& after_block = builder.add_block(loop.root());
    after_block.dataflow();

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::SymbolEvolution pass;
    bool applied = pass.run(builder, analysis_manager);

    EXPECT_FALSE(applied);
}

// Test: Multiple writes - Negative Case
// Symbol is written multiple times in the loop
TEST(SymbolEvolutionTest, MultipleWrites_Negative) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    builder.add_container("x", desc);
    auto i = symbolic::symbol("i");
    auto x = symbolic::symbol("x");

    auto& root = builder.subject().root();

    // x = 0
    auto& init_block = builder.add_block(root);
    init_block.dataflow();

    builder.add_block_after(root, init_block, {{x, symbolic::zero()}}, init_block.debug_info());

    // for (i = 0; i < 10; i++) { x = x + 1; x = x + 2; }
    auto& loop =
        builder
            .add_for(root, i, symbolic::Lt(i, symbolic::integer(10)), symbolic::zero(), symbolic::add(i, symbolic::one()));

    auto& loop_block1 = builder.add_block(loop.root());
    loop_block1.dataflow();

    builder
        .add_block_after(loop.root(), loop_block1, {{x, symbolic::add(x, symbolic::one())}}, loop_block1.debug_info());

    auto& loop_block2 = builder.add_block(loop.root());
    loop_block2.dataflow();

    builder.add_block_after(
        loop.root(), loop_block2, {{x, symbolic::add(x, symbolic::integer(2))}}, loop_block2.debug_info()
    );

    auto& after_block = builder.add_block(loop.root());
    after_block.dataflow();

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::SymbolEvolution pass;
    bool applied = pass.run(builder, analysis_manager);

    EXPECT_FALSE(applied);
}

// Test: Nested loop - Negative Case
// Symbol update is in a nested loop
TEST(SymbolEvolutionTest, NestedLoop_Negative) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    builder.add_container("j", desc);
    builder.add_container("sum", desc);
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto sum = symbolic::symbol("sum");

    auto& root = builder.subject().root();

    // sum = 0
    auto& init_block = builder.add_block(root);
    init_block.dataflow();

    builder.add_block_after(root, init_block, {{sum, symbolic::zero()}}, init_block.debug_info());

    // for (i = 0; i < 10; i++) { for (j = 0; j < 5; j++) { sum = sum + 1; } }
    auto& outer_loop =
        builder
            .add_for(root, i, symbolic::Lt(i, symbolic::integer(10)), symbolic::zero(), symbolic::add(i, symbolic::one()));

    auto& inner_loop = builder.add_for(
        outer_loop.root(), j, symbolic::Lt(j, symbolic::integer(5)), symbolic::zero(), symbolic::add(j, symbolic::one())
    );

    auto& inner_block = builder.add_block(inner_loop.root());
    inner_block.dataflow();

    builder.add_block_after(
        inner_loop.root(), inner_block, {{sum, symbolic::add(sum, symbolic::one())}}, inner_block.debug_info()
    );

    auto& after_block = builder.add_block(inner_loop.root());
    after_block.dataflow();

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::SymbolEvolution pass;
    bool applied = pass.run(builder, analysis_manager);

    EXPECT_FALSE(applied);
}
