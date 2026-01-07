#include "sdfg/passes/symbolic/symbol_evolution.h"

#include <gtest/gtest.h>

#include <ostream>

#include "sdfg/builder/structured_sdfg_builder.h"

using namespace sdfg;

// ============================================================================
// Pattern 4: Affine Update Tests with Closed-Form Verification
// Symbol increases/decreases by a constant each iteration
// ============================================================================

// Positive case: sum = sum + 5, init = 0
// Expected closed-form: sum = 5*i
TEST(SymbolEvolutionTest, Pattern4_AffineUpdate_Addition) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    builder.add_container("sum", desc);
    auto i = symbolic::symbol("i");
    auto sum = symbolic::symbol("sum");

    auto& root = builder.subject().root();

    // Initialize: sum = 0
    auto& init_block = builder.add_block(root);
    builder.add_block_after(root, init_block, {{sum, symbolic::zero()}}, init_block.debug_info());

    // Loop: for (i = 0; i < 10; i++) { sum = sum + 5; }
    auto& loop =
        builder
            .add_for(root, i, symbolic::Lt(i, symbolic::integer(10)), symbolic::zero(), symbolic::add(i, symbolic::one()));

    auto& loop_block = builder.add_block(loop.root());
    builder.add_block_after(
        loop.root(), loop_block, {{sum, symbolic::add(sum, symbolic::integer(5))}}, loop_block.debug_info()
    );

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::SymbolEvolution pass;
    bool applied = pass.run(builder, analysis_manager);

    EXPECT_TRUE(applied);

    // Verify the closed-form solution is: sum = 5*i
    auto first_elem = loop.root().at(0);
    auto& assignments = first_elem.second.assignments();

    EXPECT_TRUE(assignments.find(sum) != assignments.end());
    if (assignments.find(sum) != assignments.end()) {
        auto evolved_expr = assignments.at(sum);
        // Expected: 5*i (since init is 0 and stride is 1)
        auto expected = symbolic::mul(symbolic::integer(5), i);
        EXPECT_TRUE(symbolic::eq(evolved_expr, expected));
    }
}

// Positive case: count = count - 3, init = 100
// Expected closed-form: count = 100 - 3*i
TEST(SymbolEvolutionTest, Pattern4_AffineUpdate_Subtraction) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    builder.add_container("count", desc);
    auto i = symbolic::symbol("i");
    auto count = symbolic::symbol("count");

    auto& root = builder.subject().root();

    // Initialize: count = 100
    auto& init_block = builder.add_block(root);
    builder.add_block_after(root, init_block, {{count, symbolic::integer(100)}}, init_block.debug_info());

    // Loop: for (i = 0; i < 10; i++) { count = count - 3; }
    auto& loop =
        builder
            .add_for(root, i, symbolic::Lt(i, symbolic::integer(10)), symbolic::zero(), symbolic::add(i, symbolic::one()));

    auto& loop_block = builder.add_block(loop.root());
    builder.add_block_after(
        loop.root(), loop_block, {{count, symbolic::sub(count, symbolic::integer(3))}}, loop_block.debug_info()
    );

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::SymbolEvolution pass;
    bool applied = pass.run(builder, analysis_manager);

    EXPECT_TRUE(applied);

    // Verify the closed-form solution is: count = 100 - 3*i
    auto first_elem = loop.root().at(0);
    auto& assignments = first_elem.second.assignments();

    EXPECT_TRUE(assignments.find(count) != assignments.end());
    if (assignments.find(count) != assignments.end()) {
        auto evolved_expr = assignments.at(count);
        // Expected: 100 - 3*i
        auto expected = symbolic::sub(symbolic::integer(100), symbolic::mul(symbolic::integer(3), i));
        EXPECT_TRUE(symbolic::eq(evolved_expr, expected));
    }
}

// Negative case: prod = prod * 2 (multiplicative, not affine)
TEST(SymbolEvolutionTest, Pattern4_AffineUpdate_Negative_Multiplication) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    builder.add_container("prod", desc);
    auto i = symbolic::symbol("i");
    auto prod = symbolic::symbol("prod");

    auto& root = builder.subject().root();

    // Initialize: prod = 1
    auto& init_block = builder.add_block(root);
    builder.add_block_after(root, init_block, {{prod, symbolic::one()}}, init_block.debug_info());

    // Loop: for (i = 0; i < 10; i++) { prod = prod * 2; }
    auto& loop =
        builder
            .add_for(root, i, symbolic::Lt(i, symbolic::integer(10)), symbolic::zero(), symbolic::add(i, symbolic::one()));

    auto& loop_block = builder.add_block(loop.root());
    builder.add_block_after(
        loop.root(), loop_block, {{prod, symbolic::mul(prod, symbolic::integer(2))}}, loop_block.debug_info()
    );

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::SymbolEvolution pass;
    bool applied = pass.run(builder, analysis_manager);

    EXPECT_FALSE(applied);
}

// ============================================================================
// Pattern 1: Constant with Verification
// ============================================================================

// Positive case: c = 42 throughout
// Expected: c = 42
// TODO: Pattern 1 is not fully supported yet - needs investigation
TEST(SymbolEvolutionTest, DISABLED_Pattern1_Constant) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    builder.add_container("c", desc);
    auto i = symbolic::symbol("i");
    auto c = symbolic::symbol("c");

    auto& root = builder.subject().root();

    // Initialize: c = 42
    auto& init_block = builder.add_block(root);
    builder.add_block_after(root, init_block, {{c, symbolic::integer(42)}}, init_block.debug_info());

    // Loop: for (i = 0; i < 10; i++) { c = 42; }
    auto& loop =
        builder
            .add_for(root, i, symbolic::Lt(i, symbolic::integer(10)), symbolic::zero(), symbolic::add(i, symbolic::one()));

    auto& loop_block = builder.add_block(loop.root());
    builder.add_block_after(loop.root(), loop_block, {{c, symbolic::integer(42)}}, loop_block.debug_info());

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::SymbolEvolution pass;
    bool applied = pass.run(builder, analysis_manager);

    EXPECT_TRUE(applied);

    // Verify the closed-form solution: c = 42
    auto first_elem = loop.root().at(0);
    auto& assignments = first_elem.second.assignments();

    EXPECT_TRUE(assignments.find(c) != assignments.end());
    if (assignments.find(c) != assignments.end()) {
        auto evolved_expr = assignments.at(c);
        auto expected = symbolic::integer(42);
        EXPECT_TRUE(symbolic::eq(evolved_expr, expected));
    }
}

// ============================================================================
// Pattern 2: Loop Alias with Verification
// ============================================================================

// Positive case: j tracks i exactly
// Expected: j = i
// TODO: Pattern 2 is not fully supported yet - needs investigation
TEST(SymbolEvolutionTest, DISABLED_Pattern2_LoopAlias) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    builder.add_container("j", desc);
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");

    auto& root = builder.subject().root();

    // Initialize: j = 0 (same as i)
    auto& init_block = builder.add_block(root);
    builder.add_block_after(root, init_block, {{j, symbolic::zero()}}, init_block.debug_info());

    // Loop: for (i = 0; i < 10; i++) { j = i + 1; }
    auto& loop =
        builder
            .add_for(root, i, symbolic::Lt(i, symbolic::integer(10)), symbolic::zero(), symbolic::add(i, symbolic::one()));

    auto& loop_block = builder.add_block(loop.root());
    builder.add_block_after(loop.root(), loop_block, {{j, symbolic::add(i, symbolic::one())}}, loop_block.debug_info());

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::SymbolEvolution pass;
    bool applied = pass.run(builder, analysis_manager);

    EXPECT_TRUE(applied);

    // Verify the closed-form solution: j = i
    auto first_elem = loop.root().at(0);
    auto& assignments = first_elem.second.assignments();

    EXPECT_TRUE(assignments.find(j) != assignments.end());
    if (assignments.find(j) != assignments.end()) {
        auto evolved_expr = assignments.at(j);
        auto expected = i;
        EXPECT_TRUE(symbolic::eq(evolved_expr, expected));
    }
}

// ============================================================================
// Negative Test Cases - Edge Cases
// ============================================================================

// Symbol is used after being updated
TEST(SymbolEvolutionTest, Negative_UsedAfterUpdate) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    builder.add_container("sum", desc);
    builder.add_container("temp", desc);
    auto i = symbolic::symbol("i");
    auto sum = symbolic::symbol("sum");
    auto temp = symbolic::symbol("temp");

    auto& root = builder.subject().root();

    // Initialize: sum = 0
    auto& init_block = builder.add_block(root);
    builder.add_block_after(root, init_block, {{sum, symbolic::zero()}}, init_block.debug_info());

    // Loop: for (i = 0; i < 10; i++) { sum = sum + 5; temp = sum; }
    auto& loop =
        builder
            .add_for(root, i, symbolic::Lt(i, symbolic::integer(10)), symbolic::zero(), symbolic::add(i, symbolic::one()));

    auto& loop_block1 = builder.add_block(loop.root());
    builder.add_block_after(
        loop.root(), loop_block1, {{sum, symbolic::add(sum, symbolic::integer(5))}}, loop_block1.debug_info()
    );

    auto& loop_block2 = builder.add_block(loop.root());
    builder.add_block_after(loop.root(), loop_block2, {{temp, sum}}, loop_block2.debug_info());

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::SymbolEvolution pass;
    bool applied = pass.run(builder, analysis_manager);

    EXPECT_FALSE(applied);
}

// Symbol is written multiple times
TEST(SymbolEvolutionTest, Negative_MultipleWrites) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    builder.add_container("x", desc);
    auto i = symbolic::symbol("i");
    auto x = symbolic::symbol("x");

    auto& root = builder.subject().root();

    // Initialize: x = 0
    auto& init_block = builder.add_block(root);
    builder.add_block_after(root, init_block, {{x, symbolic::zero()}}, init_block.debug_info());

    // Loop: for (i = 0; i < 10; i++) { x = x + 1; x = x + 2; }
    auto& loop =
        builder
            .add_for(root, i, symbolic::Lt(i, symbolic::integer(10)), symbolic::zero(), symbolic::add(i, symbolic::one()));

    auto& loop_block1 = builder.add_block(loop.root());
    builder
        .add_block_after(loop.root(), loop_block1, {{x, symbolic::add(x, symbolic::one())}}, loop_block1.debug_info());

    auto& loop_block2 = builder.add_block(loop.root());
    builder.add_block_after(
        loop.root(), loop_block2, {{x, symbolic::add(x, symbolic::integer(2))}}, loop_block2.debug_info()
    );

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::SymbolEvolution pass;
    bool applied = pass.run(builder, analysis_manager);

    EXPECT_FALSE(applied);
}
