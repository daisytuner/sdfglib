#include "sdfg/passes/structured_control_flow/loop_normalization.h"

#include <gtest/gtest.h>

using namespace sdfg;

TEST(LoopNormalizationTest, Unequality_StrideOne_Positive) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc_unsigned(types::PrimitiveType::UInt64);
    types::Scalar desc_signed(types::PrimitiveType::Int64);
    builder.add_container("N", desc_unsigned, true);
    builder.add_container("i", desc_signed);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Ne(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::one());

    auto& loop = builder.add_for(root, indvar, condition, init, update);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::LoopNormalization pass;
    pass.run(builder, analysis_manager);

    // Check
    EXPECT_TRUE(SymEngine::eq(*loop.condition(), *symbolic::Lt(indvar, bound)));
}

TEST(LoopNormalizationTest, Unequality_StrideOne_Negative) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc_unsigned(types::PrimitiveType::UInt64);
    types::Scalar desc_signed(types::PrimitiveType::Int64);
    builder.add_container("N", desc_unsigned, true);
    builder.add_container("i", desc_signed);
    builder.add_container("i_ext", desc_signed, true);

    // Define loop
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::symbol("N");
    auto condition = symbolic::Ne(indvar, symbolic::zero());
    auto update = symbolic::sub(indvar, symbolic::one());

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& seq = builder.add_sequence(loop.root(), {{symbolic::symbol("i_ext"), indvar}});

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::LoopNormalization pass;
    pass.run(builder, analysis_manager);

    // Check
    EXPECT_TRUE(SymEngine::eq(*loop.condition(), *symbolic::Lt(indvar, symbolic::add(init, symbolic::one()))));

    auto& assignments = loop.root().at(0).second.assignments();
    EXPECT_TRUE(symbolic::
                    eq(assignments.at(symbolic::symbol("i_ext")),
                       symbolic::sub(init, symbolic::sub(indvar, symbolic::one()))));
}

TEST(LoopNormalizationTest, Unequality_StrideNotOne_Positive) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc_unsigned(types::PrimitiveType::UInt64);
    types::Scalar desc_signed(types::PrimitiveType::Int64);
    builder.add_container("N", desc_unsigned, true);
    builder.add_container("i", desc_signed);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Ne(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(2));

    auto& loop = builder.add_for(root, indvar, condition, init, update);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::LoopNormalization pass;
    pass.run(builder, analysis_manager);

    // Check
    EXPECT_TRUE(SymEngine::eq(*loop.condition(), *symbolic::Ne(indvar, bound)));
    // EXPECT_TRUE(SymEngine::eq(*loop.condition(), *symbolic::Lt(indvar, bound)));
}

TEST(LoopNormalizationTest, Unequality_StrideNotOne_Negative) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc_unsigned(types::PrimitiveType::UInt64);
    types::Scalar desc_signed(types::PrimitiveType::Int64);
    builder.add_container("N", desc_unsigned, true);
    builder.add_container("i", desc_signed);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Ne(indvar, bound);
    auto update = symbolic::sub(indvar, symbolic::integer(2));

    auto& loop = builder.add_for(root, indvar, condition, init, update);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::LoopNormalization pass;
    pass.run(builder, analysis_manager);

    // Check
    EXPECT_TRUE(SymEngine::eq(*loop.condition(), *symbolic::Ne(indvar, bound)));
    // EXPECT_TRUE(SymEngine::eq(*loop.condition(), *symbolic::Lt(indvar, bound)));
}

TEST(LoopNormalizationTest, AndUnequality) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc_unsigned(types::PrimitiveType::UInt64);
    types::Scalar desc_signed(types::PrimitiveType::Int64);
    builder.add_container("N", desc_unsigned, true);
    builder.add_container("M", desc_unsigned, true);
    builder.add_container("i", desc_signed);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto bound2 = symbolic::symbol("M");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::And(symbolic::Ne(indvar, bound), symbolic::Ne(indvar, bound2));
    auto update = symbolic::add(indvar, symbolic::one());

    auto& loop = builder.add_for(root, indvar, condition, init, update);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::LoopNormalization pass;
    pass.run(builder, analysis_manager);

    // Check
    EXPECT_TRUE(SymEngine::eq(*loop.condition(), *symbolic::And(symbolic::Lt(indvar, bound), symbolic::Lt(indvar, bound2)))
    );
}

TEST(LoopNormalizationTest, OrUnequality) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc_unsigned(types::PrimitiveType::UInt64);
    types::Scalar desc_signed(types::PrimitiveType::Int64);
    builder.add_container("N", desc_unsigned, true);
    builder.add_container("M", desc_unsigned, true);
    builder.add_container("i", desc_signed);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto bound2 = symbolic::symbol("M");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Or(symbolic::Ne(indvar, bound), symbolic::Ne(indvar, bound2));
    auto update = symbolic::add(indvar, symbolic::one());

    auto& loop = builder.add_for(root, indvar, condition, init, update);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::LoopNormalization pass;
    pass.run(builder, analysis_manager);

    // Check
    EXPECT_TRUE(SymEngine::eq(*loop.condition(), *symbolic::Or(symbolic::Lt(indvar, bound), symbolic::Lt(indvar, bound2)))
    );
}
