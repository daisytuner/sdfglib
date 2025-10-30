#include "sdfg/passes/structured_control_flow/loop_normalization.h"

#include "sdfg/data_flow/library_nodes/math/intrinsic.h"

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
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    // Check
    // EXPECT_TRUE(SymEngine::eq(*loop.condition(), *symbolic::Lt(indvar, symbolic::add(init, symbolic::one()))));

    // auto& assignments = loop.root().at(0).second.assignments();
    // EXPECT_TRUE(symbolic::
    //                 eq(assignments.at(symbolic::symbol("i_ext")),
    //                    symbolic::sub(init, symbolic::sub(indvar, symbolic::one()))));
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
    EXPECT_FALSE(pass.run(builder, analysis_manager));

    // Check
    EXPECT_TRUE(SymEngine::eq(*loop.condition(), *symbolic::Ne(indvar, bound)));
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

TEST(LoopNormalizationTest, Rotate) {
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
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::sub(symbolic::symbol("N"), symbolic::one());
    auto condition = symbolic::Lt(symbolic::zero(), indvar);
    auto update = symbolic::sub(indvar, symbolic::one());

    auto& loop = builder.add_for(root, indvar, condition, init, update);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::LoopNormalization pass;
    EXPECT_FALSE(pass.run(builder, analysis_manager));

    // Check
    // EXPECT_TRUE(symbolic::eq(loop.condition(), symbolic::Lt(indvar, symbolic::symbol("N"))));
}

TEST(LoopNormalizationTest, Rotate_NonCommutative) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc_signed(types::PrimitiveType::Int64);
    builder.add_container("N", desc_signed, true);
    builder.add_container("M", desc_signed, true);
    builder.add_container("i", desc_signed);
    types::Scalar desc_double(types::PrimitiveType::Double);
    builder.add_container("d", desc_double);
    builder.add_container("tmp", desc_double);
    builder.add_container("tmp2", desc_double);

    // Define loop
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::sub(symbolic::symbol("N"), symbolic::one());
    auto condition = symbolic::Lt(symbolic::zero(), indvar);
    auto update = symbolic::sub(indvar, symbolic::one());

    auto& loop = builder.add_for(root, indvar, condition, init, update);

    // cast indvar to double
    {
        auto& block = builder.add_block(loop.root());
        auto& node_in = builder.add_access(block, "i");
        auto& node_out = builder.add_access(block, "tmp");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
        builder.add_computational_memlet(block, node_in, tasklet, "_in", {});
        builder.add_computational_memlet(block, tasklet, "_out", node_out, {});
    }
    // tmp2 = d + tmp
    {
        auto& block = builder.add_block(loop.root());
        auto& node_in1 = builder.add_access(block, "tmp");
        auto& node_in2 = builder.add_access(block, "d");
        auto& node_out = builder.add_access(block, "tmp2");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, {"_out"}, {"_in1", "_in2"});
        builder.add_computational_memlet(block, node_in1, tasklet, "_in1", {});
        builder.add_computational_memlet(block, node_in2, tasklet, "_in2", {});
        builder.add_computational_memlet(block, tasklet, "_out", node_out, {});
    }
    // Non-commutative: d = sin(d)
    {
        auto& block = builder.add_block(loop.root());
        auto& node_in = builder.add_access(block, "d");
        auto& node_out = builder.add_access(block, "d");
        auto& tasklet = builder.add_library_node<sdfg::math::IntrinsicNode>(block, DebugInfo(), "sin", 1);
        builder.add_computational_memlet(block, node_in, tasklet, "_in1", {}, desc_double);
        builder.add_computational_memlet(block, tasklet, "_out", node_out, {}, desc_double);
    }

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::LoopNormalization pass;
    EXPECT_FALSE(pass.run(builder, analysis_manager));
}
