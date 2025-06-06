#include "sdfg/analysis/assumptions_analysis.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"

using namespace sdfg;

TEST(SymbolsTest, Init_bool) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType::CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc(types::PrimitiveType::Bool);
    builder.add_container("N", desc, true);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto& assumptions = analysis.get(root);

    // Check
    EXPECT_TRUE(
        SymEngine::eq(*assumptions.at(symbolic::symbol("N")).lower_bound(), *symbolic::integer(0)));
    EXPECT_TRUE(
        SymEngine::eq(*assumptions.at(symbolic::symbol("N")).upper_bound(), *symbolic::integer(1)));
}

TEST(SymbolsTest, Init_i8) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType::CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc_unsigned(types::PrimitiveType::UInt8);
    types::Scalar desc_signed(types::PrimitiveType::Int8);
    builder.add_container("N", desc_unsigned, true);
    builder.add_container("M", desc_signed, true);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto& assumptions = analysis.get(root);

    // Check
    EXPECT_TRUE(
        SymEngine::eq(*assumptions.at(symbolic::symbol("N")).lower_bound(), *symbolic::integer(0)));
    EXPECT_TRUE(SymEngine::eq(*assumptions.at(symbolic::symbol("N")).upper_bound(),
                              *symbolic::integer(255)));
    EXPECT_TRUE(SymEngine::eq(*assumptions.at(symbolic::symbol("M")).lower_bound(),
                              *symbolic::integer(-128)));
    EXPECT_TRUE(SymEngine::eq(*assumptions.at(symbolic::symbol("M")).upper_bound(),
                              *symbolic::integer(127)));
}

TEST(SymbolsTest, Init_i16) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType::CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc_unsigned(types::PrimitiveType::UInt16);
    types::Scalar desc_signed(types::PrimitiveType::Int16);
    builder.add_container("N", desc_unsigned, true);
    builder.add_container("M", desc_signed, true);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto& assumptions = analysis.get(root);

    // Check
    EXPECT_TRUE(
        SymEngine::eq(*assumptions.at(symbolic::symbol("N")).lower_bound(), *symbolic::integer(0)));
    EXPECT_TRUE(SymEngine::eq(*assumptions.at(symbolic::symbol("N")).upper_bound(),
                              *symbolic::integer(65535)));
    EXPECT_TRUE(SymEngine::eq(*assumptions.at(symbolic::symbol("M")).lower_bound(),
                              *symbolic::integer(-32768)));
    EXPECT_TRUE(SymEngine::eq(*assumptions.at(symbolic::symbol("M")).upper_bound(),
                              *symbolic::integer(32767)));
}

TEST(SymbolsTest, Init_i32) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType::CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc_unsigned(types::PrimitiveType::UInt32);
    types::Scalar desc_signed(types::PrimitiveType::Int32);
    builder.add_container("N", desc_unsigned, true);
    builder.add_container("M", desc_signed, true);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto& assumptions = analysis.get(root);

    // Check
    EXPECT_TRUE(
        SymEngine::eq(*assumptions.at(symbolic::symbol("N")).lower_bound(), *symbolic::integer(0)));
    EXPECT_TRUE(SymEngine::eq(*assumptions.at(symbolic::symbol("N")).upper_bound(),
                              *symbolic::integer(4294967295)));
    EXPECT_TRUE(SymEngine::eq(*assumptions.at(symbolic::symbol("M")).lower_bound(),
                              *symbolic::integer(-2147483648)));
    EXPECT_TRUE(SymEngine::eq(*assumptions.at(symbolic::symbol("M")).upper_bound(),
                              *symbolic::integer(2147483647)));
}

TEST(SymbolsTest, Init_i64) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType::CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc_unsigned(types::PrimitiveType::UInt64);
    types::Scalar desc_signed(types::PrimitiveType::Int64);
    builder.add_container("N", desc_unsigned, true);
    builder.add_container("M", desc_signed, true);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto& assumptions = analysis.get(root);

    // Check
    EXPECT_TRUE(
        SymEngine::eq(*assumptions.at(symbolic::symbol("N")).lower_bound(), *symbolic::integer(0)));
    EXPECT_TRUE(
        SymEngine::eq(*assumptions.at(symbolic::symbol("N")).upper_bound(), *symbolic::infty(1)));
    EXPECT_TRUE(
        SymEngine::eq(*assumptions.at(symbolic::symbol("M")).lower_bound(), *symbolic::infty(-1)));
    EXPECT_TRUE(
        SymEngine::eq(*assumptions.at(symbolic::symbol("M")).upper_bound(), *symbolic::infty(1)));
}

TEST(SymbolsTest, Iteration_Rectangle_1D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType::CPU);

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
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto& assumptions = analysis.get(loop.root());

    // Check
    EXPECT_TRUE(
        SymEngine::eq(*assumptions.at(symbolic::symbol("N")).lower_bound(), *symbolic::integer(0)));
    EXPECT_TRUE(
        SymEngine::eq(*assumptions.at(symbolic::symbol("N")).upper_bound(), *symbolic::infty(1)));
    EXPECT_TRUE(
        SymEngine::eq(*assumptions.at(symbolic::symbol("i")).lower_bound(), *symbolic::integer(0)));
    EXPECT_TRUE(SymEngine::eq(*assumptions.at(symbolic::symbol("i")).upper_bound(),
                              *symbolic::sub(symbolic::symbol("N"), symbolic::integer(1))));
}

TEST(SymbolsTest, Iteration_Rectangle_1D_And) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType::CPU);

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
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition =
        symbolic::And(symbolic::Le(indvar, bound), symbolic::Le(indvar, symbolic::symbol("M")));
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto& assumptions = analysis.get(loop.root());

    // Check
    EXPECT_TRUE(
        SymEngine::eq(*assumptions.at(symbolic::symbol("N")).lower_bound(), *symbolic::integer(0)));
    EXPECT_TRUE(
        SymEngine::eq(*assumptions.at(symbolic::symbol("N")).upper_bound(), *symbolic::infty(1)));
    EXPECT_TRUE(
        SymEngine::eq(*assumptions.at(symbolic::symbol("M")).lower_bound(), *symbolic::integer(0)));
    EXPECT_TRUE(
        SymEngine::eq(*assumptions.at(symbolic::symbol("M")).upper_bound(), *symbolic::infty(1)));
    EXPECT_TRUE(
        SymEngine::eq(*assumptions.at(symbolic::symbol("i")).lower_bound(), *symbolic::integer(0)));
    EXPECT_TRUE(SymEngine::eq(*assumptions.at(symbolic::symbol("i")).upper_bound(),
                              *symbolic::min(symbolic::symbol("N"), symbolic::symbol("M"))));
}

TEST(SymbolsTest, Iteration_Triangle_2D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType::CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc_unsigned(types::PrimitiveType::UInt64);
    types::Scalar desc_signed(types::PrimitiveType::Int64);
    builder.add_container("N", desc_unsigned, true);
    builder.add_container("i", desc_signed);
    builder.add_container("j", desc_signed);

    // Define loop
    auto bound = symbolic::sub(symbolic::symbol("N"), symbolic::integer(1));
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);

    auto bound_2 = symbolic::symbol("N");
    auto indvar_2 = symbolic::symbol("j");
    auto init_2 = symbolic::add(indvar, symbolic::integer(1));
    auto condition_2 = symbolic::Lt(indvar_2, bound_2);
    auto update_2 = symbolic::add(indvar_2, symbolic::integer(1));

    auto& loop2 = builder.add_for(loop.root(), indvar_2, condition_2, init_2, update_2);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto& assumptions = analysis.get(loop2.root());

    // Check
    EXPECT_TRUE(
        SymEngine::eq(*assumptions.at(symbolic::symbol("N")).lower_bound(), *symbolic::integer(0)));
    EXPECT_TRUE(
        SymEngine::eq(*assumptions.at(symbolic::symbol("N")).upper_bound(), *symbolic::infty(1)));
    EXPECT_TRUE(
        SymEngine::eq(*assumptions.at(symbolic::symbol("i")).lower_bound(), *symbolic::integer(0)));
    EXPECT_TRUE(SymEngine::eq(*assumptions.at(symbolic::symbol("i")).upper_bound(),
                              *symbolic::sub(symbolic::symbol("N"), symbolic::integer(2))));
    EXPECT_TRUE(SymEngine::eq(*assumptions.at(symbolic::symbol("j")).lower_bound(),
                              *symbolic::add(indvar, symbolic::integer(1))));
    EXPECT_TRUE(SymEngine::eq(*assumptions.at(symbolic::symbol("j")).upper_bound(),
                              *symbolic::sub(symbolic::symbol("N"), symbolic::integer(1))));
}
