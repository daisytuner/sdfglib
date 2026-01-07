#include "sdfg/passes/symbolic/symbol_evolution.h"

#include <gtest/gtest.h>

#include <ostream>

#include "sdfg/builder/structured_sdfg_builder.h"

using namespace sdfg;

// ============================================================================
// Pattern 4: Affine Update Tests
// Symbol increases/decreases by a constant each iteration
// ============================================================================

// Positive case: sum = sum + 5
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
}

// Positive case: count = count - 3
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
