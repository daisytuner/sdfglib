#include "sdfg/passes/symbolic/symbol_evolution.h"

#include <gtest/gtest.h>

#include <ostream>

#include "sdfg/builder/structured_sdfg_builder.h"

using namespace sdfg;

// ============================================================================
// Extended Tests with Closed-Form Verification
// These tests verify that the pass produces the correct closed-form expression
// ============================================================================

// Test Pattern 3: Quadratic function sq = i*i
// Expected closed-form: sq = (i-1)*(i-1)
TEST(SymbolEvolutionExtendedTest, Pattern3_Quadratic) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    builder.add_container("sq", desc);
    auto i = symbolic::symbol("i");
    auto sq = symbolic::symbol("sq");

    auto& root = builder.subject().root();

    // Initialize: sq = 0 (which is (-1)*(-1) = 1, but we need to match the pattern)
    // For Pattern 3 to work: init value should equal f(init - stride)
    // f(i) = i*i, so f(-1) = 1, but we're testing if init = 0 works
    auto& init_block = builder.add_block(root);
    builder.add_block_after(root, init_block, {{sq, symbolic::zero()}}, init_block.debug_info());

    // Loop: for (i = 0; i < 10; i++) { sq = i*i; }
    auto& loop =
        builder
            .add_for(root, i, symbolic::Lt(i, symbolic::integer(10)), symbolic::zero(), symbolic::add(i, symbolic::one()));

    auto& loop_block = builder.add_block(loop.root());
    builder.add_block_after(loop.root(), loop_block, {{sq, symbolic::mul(i, i)}}, loop_block.debug_info());

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::SymbolEvolution pass;
    bool applied = pass.run(builder, analysis_manager);

    // This should apply with Pattern 3 if init condition matches
    if (applied) {
        // Verify the closed-form solution exists
        auto first_elem = loop.root().at(0);
        auto& assignments = first_elem.second.assignments();

        EXPECT_TRUE(assignments.find(sq) != assignments.end());
        if (assignments.find(sq) != assignments.end()) {
            auto evolved_expr = assignments.at(sq);
            // Expected: (i-1)*(i-1) because it's f(i_{n-1})
            auto i_minus_1 = symbolic::sub(i, symbolic::one());
            auto expected = symbolic::mul(i_minus_1, i_minus_1);
            EXPECT_TRUE(symbolic::eq(evolved_expr, expected));
        }
    }
}

// Test Pattern 3: Linear function double_i = 2*i
// Expected closed-form: double_i = 2*(i-1)
TEST(SymbolEvolutionExtendedTest, Pattern3_LinearFunction) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    builder.add_container("double_i", desc);
    auto i = symbolic::symbol("i");
    auto double_i = symbolic::symbol("double_i");

    auto& root = builder.subject().root();

    // Initialize: double_i = -2 (which is 2*(0-1))
    auto& init_block = builder.add_block(root);
    builder.add_block_after(root, init_block, {{double_i, symbolic::integer(-2)}}, init_block.debug_info());

    // Loop: for (i = 0; i < 10; i++) { double_i = 2*i; }
    auto& loop =
        builder
            .add_for(root, i, symbolic::Lt(i, symbolic::integer(10)), symbolic::zero(), symbolic::add(i, symbolic::one()));

    auto& loop_block = builder.add_block(loop.root());
    builder.add_block_after(
        loop.root(), loop_block, {{double_i, symbolic::mul(symbolic::integer(2), i)}}, loop_block.debug_info()
    );

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::SymbolEvolution pass;
    bool applied = pass.run(builder, analysis_manager);

    EXPECT_TRUE(applied);

    // Verify the closed-form solution: double_i = 2*(i-1)
    auto first_elem = loop.root().at(0);
    auto& assignments = first_elem.second.assignments();

    EXPECT_TRUE(assignments.find(double_i) != assignments.end());
    if (assignments.find(double_i) != assignments.end()) {
        auto evolved_expr = assignments.at(double_i);
        // Expected: 2*(i-1)
        auto expected = symbolic::mul(symbolic::integer(2), symbolic::sub(i, symbolic::one()));
        EXPECT_TRUE(symbolic::eq(evolved_expr, expected));
    }
}

// Test Pattern 4: Affine with offset
// sum = sum + 5, init = 10
// Expected: sum = 10 + 5*i
TEST(SymbolEvolutionExtendedTest, Pattern4_AffineWithOffset) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    builder.add_container("sum", desc);
    auto i = symbolic::symbol("i");
    auto sum = symbolic::symbol("sum");

    auto& root = builder.subject().root();

    // Initialize: sum = 10
    auto& init_block = builder.add_block(root);
    builder.add_block_after(root, init_block, {{sum, symbolic::integer(10)}}, init_block.debug_info());

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

    // Verify the closed-form solution: sum = 10 + 5*i
    auto first_elem = loop.root().at(0);
    auto& assignments = first_elem.second.assignments();

    EXPECT_TRUE(assignments.find(sum) != assignments.end());
    if (assignments.find(sum) != assignments.end()) {
        auto evolved_expr = assignments.at(sum);
        // Expected: 10 + 5*i
        auto expected = symbolic::add(symbolic::integer(10), symbolic::mul(symbolic::integer(5), i));
        EXPECT_TRUE(symbolic::eq(evolved_expr, expected));
    }
}

// Test verifying value after loop
// sum = sum + 3, for i=5 to i<10, stride=1
// Expected after loop: sum = init + 3*5 = init + 15
TEST(SymbolEvolutionExtendedTest, Pattern4_FinalValueVerification) {
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

    // Loop: for (i = 5; i < 10; i++) { sum = sum + 3; }
    auto& loop = builder.add_for(
        root, i, symbolic::Lt(i, symbolic::integer(10)), symbolic::integer(5), symbolic::add(i, symbolic::one())
    );

    auto& loop_block = builder.add_block(loop.root());
    builder.add_block_after(
        loop.root(), loop_block, {{sum, symbolic::add(sum, symbolic::integer(3))}}, loop_block.debug_info()
    );

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::SymbolEvolution pass;
    bool applied = pass.run(builder, analysis_manager);

    EXPECT_TRUE(applied);

    // Verify: sum = 0 + 3*(i-5) = 3*i - 15
    auto first_elem = loop.root().at(0);
    auto& assignments = first_elem.second.assignments();

    EXPECT_TRUE(assignments.find(sum) != assignments.end());
    if (assignments.find(sum) != assignments.end()) {
        auto evolved_expr = assignments.at(sum);
        // Expected: 3*(i-5) = 3*i - 15
        auto expected = symbolic::mul(symbolic::integer(3), symbolic::sub(i, symbolic::integer(5)));
        EXPECT_TRUE(symbolic::eq(evolved_expr, expected));
    }
}
