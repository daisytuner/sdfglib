#include "sdfg/analysis/assumptions_analysis.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/array.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"

using namespace sdfg;

TEST(AssumptionsAnalysisTest, Init_bool) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc(types::PrimitiveType::Bool);
    builder.add_container("N", desc, true);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto& assumptions = analysis.get(root, true);

    // Check
    EXPECT_TRUE(SymEngine::eq(*assumptions.at(symbolic::symbol("N")).lower_bound(), *symbolic::integer(0)));
    EXPECT_TRUE(SymEngine::eq(*assumptions.at(symbolic::symbol("N")).upper_bound(), *symbolic::integer(1)));
    EXPECT_TRUE(assumptions.at(symbolic::symbol("N")).tight_lower_bound().is_null());
    EXPECT_TRUE(assumptions.at(symbolic::symbol("N")).tight_upper_bound().is_null());
    EXPECT_EQ(analysis.parameters().size(), 1);
    EXPECT_TRUE(analysis.is_parameter("N"));
}

TEST(AssumptionsAnalysisTest, Init_i8) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

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
    auto& assumptions = analysis.get(root, true);

    // Check
    EXPECT_TRUE(SymEngine::eq(*assumptions.at(symbolic::symbol("N")).lower_bound(), *symbolic::integer(0)));
    EXPECT_TRUE(SymEngine::eq(*assumptions.at(symbolic::symbol("N")).upper_bound(), *symbolic::integer(255)));
    EXPECT_TRUE(SymEngine::eq(*assumptions.at(symbolic::symbol("M")).lower_bound(), *symbolic::integer(-128)));
    EXPECT_TRUE(SymEngine::eq(*assumptions.at(symbolic::symbol("M")).upper_bound(), *symbolic::integer(127)));
    EXPECT_TRUE(assumptions.at(symbolic::symbol("N")).tight_lower_bound().is_null());
    EXPECT_TRUE(assumptions.at(symbolic::symbol("N")).tight_upper_bound().is_null());
    EXPECT_TRUE(assumptions.at(symbolic::symbol("M")).tight_lower_bound().is_null());
    EXPECT_TRUE(assumptions.at(symbolic::symbol("M")).tight_upper_bound().is_null());
    EXPECT_EQ(analysis.parameters().size(), 2);
    EXPECT_TRUE(analysis.is_parameter("N"));
    EXPECT_TRUE(analysis.is_parameter("M"));
}

TEST(AssumptionsAnalysisTest, Init_i16) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

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
    auto& assumptions = analysis.get(root, true);

    // Check
    EXPECT_TRUE(SymEngine::eq(*assumptions.at(symbolic::symbol("N")).lower_bound(), *symbolic::integer(0)));
    EXPECT_TRUE(SymEngine::eq(*assumptions.at(symbolic::symbol("N")).upper_bound(), *symbolic::integer(65535)));
    EXPECT_TRUE(SymEngine::eq(*assumptions.at(symbolic::symbol("M")).lower_bound(), *symbolic::integer(-32768)));
    EXPECT_TRUE(SymEngine::eq(*assumptions.at(symbolic::symbol("M")).upper_bound(), *symbolic::integer(32767)));
    EXPECT_TRUE(assumptions.at(symbolic::symbol("N")).tight_lower_bound().is_null());
    EXPECT_TRUE(assumptions.at(symbolic::symbol("N")).tight_upper_bound().is_null());
    EXPECT_TRUE(assumptions.at(symbolic::symbol("M")).tight_lower_bound().is_null());
    EXPECT_TRUE(assumptions.at(symbolic::symbol("M")).tight_upper_bound().is_null());
    EXPECT_EQ(analysis.parameters().size(), 2);
    EXPECT_TRUE(analysis.is_parameter("N"));
    EXPECT_TRUE(analysis.is_parameter("M"));
}

TEST(AssumptionsAnalysisTest, Init_i32) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

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
    auto& assumptions = analysis.get(root, true);

    // Check
    EXPECT_TRUE(SymEngine::eq(*assumptions.at(symbolic::symbol("N")).lower_bound(), *symbolic::integer(0)));
    EXPECT_TRUE(SymEngine::eq(*assumptions.at(symbolic::symbol("N")).upper_bound(), *symbolic::integer(4294967295)));
    EXPECT_TRUE(SymEngine::eq(*assumptions.at(symbolic::symbol("M")).lower_bound(), *symbolic::integer(-2147483648)));
    EXPECT_TRUE(SymEngine::eq(*assumptions.at(symbolic::symbol("M")).upper_bound(), *symbolic::integer(2147483647)));
    EXPECT_TRUE(assumptions.at(symbolic::symbol("N")).tight_lower_bound().is_null());
    EXPECT_TRUE(assumptions.at(symbolic::symbol("N")).tight_upper_bound().is_null());
    EXPECT_TRUE(assumptions.at(symbolic::symbol("M")).tight_lower_bound().is_null());
    EXPECT_TRUE(assumptions.at(symbolic::symbol("M")).tight_upper_bound().is_null());
    EXPECT_EQ(analysis.parameters().size(), 2);
    EXPECT_TRUE(analysis.is_parameter("N"));
    EXPECT_TRUE(analysis.is_parameter("M"));
}

TEST(AssumptionsAnalysisTest, Init_i64) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

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
    auto& assumptions = analysis.get(root, true);

    // Check
    EXPECT_TRUE(SymEngine::eq(*assumptions.at(symbolic::symbol("N")).lower_bound(), *symbolic::integer(0)));
    EXPECT_TRUE(SymEngine::eq(*assumptions.at(symbolic::symbol("N")).upper_bound(), *SymEngine::Inf));
    EXPECT_TRUE(SymEngine::
                    eq(*assumptions.at(symbolic::symbol("M")).lower_bound(),
                       *symbolic::integer(std::numeric_limits<int64_t>::min())));
    EXPECT_TRUE(SymEngine::
                    eq(*assumptions.at(symbolic::symbol("M")).upper_bound(),
                       *symbolic::integer(std::numeric_limits<int64_t>::max())));
    EXPECT_TRUE(assumptions.at(symbolic::symbol("N")).tight_lower_bound().is_null());
    EXPECT_TRUE(assumptions.at(symbolic::symbol("N")).tight_upper_bound().is_null());
    EXPECT_TRUE(assumptions.at(symbolic::symbol("M")).tight_lower_bound().is_null());
    EXPECT_TRUE(assumptions.at(symbolic::symbol("M")).tight_upper_bound().is_null());
    EXPECT_EQ(analysis.parameters().size(), 2);
    EXPECT_TRUE(analysis.is_parameter("N"));
    EXPECT_TRUE(analysis.is_parameter("M"));
}

TEST(AssumptionsAnalysisTest, For_1D) {
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
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto& assumptions = analysis.get(loop.root());

    // Check
    EXPECT_EQ(assumptions.size(), 2);
    EXPECT_TRUE(SymEngine::eq(*assumptions.at(symbolic::symbol("i")).lower_bound(), *symbolic::integer(0)));
    EXPECT_TRUE(SymEngine::
                    eq(*assumptions.at(symbolic::symbol("i")).upper_bound(),
                       *symbolic::sub(symbolic::symbol("N"), symbolic::integer(1))));
    EXPECT_TRUE(SymEngine::eq(*assumptions.at(symbolic::symbol("i")).tight_lower_bound(), *symbolic::integer(0)));
    EXPECT_TRUE(SymEngine::
                    eq(*assumptions.at(symbolic::symbol("i")).tight_upper_bound(),
                       *symbolic::sub(symbolic::symbol("N"), symbolic::integer(1))));
    EXPECT_TRUE(assumptions.at(symbolic::symbol("i")).constant());
    EXPECT_TRUE(symbolic::eq(assumptions.at(symbolic::symbol("N")).lower_bound(), symbolic::one()));
    EXPECT_TRUE(symbolic::eq(assumptions.at(symbolic::symbol("N")).upper_bound(), SymEngine::Inf));
    EXPECT_TRUE(assumptions.at(symbolic::symbol("N")).tight_lower_bound().is_null());
    EXPECT_TRUE(assumptions.at(symbolic::symbol("N")).tight_upper_bound().is_null());
    EXPECT_TRUE(assumptions.at(symbolic::symbol("N")).constant());
    EXPECT_EQ(analysis.parameters().size(), 1);
    EXPECT_TRUE(analysis.is_parameter("N"));
    EXPECT_FALSE(analysis.is_parameter("i"));
}

TEST(AssumptionsAnalysisTest, For_1D_And) {
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
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::And(symbolic::Le(indvar, bound), symbolic::Le(indvar, symbolic::symbol("M")));
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto& assumptions = analysis.get(loop.root());

    // Check
    EXPECT_EQ(assumptions.size(), 3);
    EXPECT_TRUE(SymEngine::eq(*assumptions.at(symbolic::symbol("i")).lower_bound(), *symbolic::integer(0)));
    EXPECT_TRUE(SymEngine::
                    eq(*assumptions.at(symbolic::symbol("i")).upper_bound(),
                       *symbolic::min(symbolic::symbol("N"), symbolic::symbol("M"))));
    EXPECT_TRUE(SymEngine::eq(*assumptions.at(symbolic::symbol("i")).tight_lower_bound(), *symbolic::integer(0)));
    EXPECT_TRUE(SymEngine::
                    eq(*assumptions.at(symbolic::symbol("i")).tight_upper_bound(),
                       *symbolic::min(symbolic::symbol("N"), symbolic::symbol("M"))));
    EXPECT_TRUE(assumptions.at(symbolic::symbol("i")).constant());
    EXPECT_TRUE(symbolic::eq(assumptions.at(symbolic::symbol("N")).lower_bound(), symbolic::one()));
    EXPECT_TRUE(symbolic::eq(assumptions.at(symbolic::symbol("N")).upper_bound(), SymEngine::Inf));
    EXPECT_TRUE(assumptions.at(symbolic::symbol("N")).tight_lower_bound().is_null());
    EXPECT_TRUE(assumptions.at(symbolic::symbol("N")).tight_upper_bound().is_null());
    EXPECT_TRUE(assumptions.at(symbolic::symbol("N")).constant());
    EXPECT_TRUE(symbolic::eq(assumptions.at(symbolic::symbol("M")).lower_bound(), symbolic::one()));
    EXPECT_TRUE(symbolic::eq(assumptions.at(symbolic::symbol("M")).upper_bound(), SymEngine::Inf));
    EXPECT_TRUE(assumptions.at(symbolic::symbol("M")).tight_lower_bound().is_null());
    EXPECT_TRUE(assumptions.at(symbolic::symbol("M")).tight_upper_bound().is_null());
    EXPECT_TRUE(assumptions.at(symbolic::symbol("M")).constant());
    EXPECT_EQ(analysis.parameters().size(), 2);
    EXPECT_TRUE(analysis.is_parameter("N"));
    EXPECT_TRUE(analysis.is_parameter("M"));
    EXPECT_FALSE(analysis.is_parameter("i"));
}

TEST(AssumptionsAnalysisTest, For_2D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

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
    EXPECT_EQ(assumptions.size(), 3);
    EXPECT_TRUE(SymEngine::eq(*assumptions.at(symbolic::symbol("i")).lower_bound(), *symbolic::integer(0)));
    EXPECT_TRUE(SymEngine::
                    eq(*assumptions.at(symbolic::symbol("i")).upper_bound(),
                       *symbolic::sub(symbolic::symbol("N"), symbolic::integer(2))));
    EXPECT_TRUE(SymEngine::
                    eq(*assumptions.at(symbolic::symbol("j")).lower_bound(),
                       *symbolic::add(indvar, symbolic::integer(1))));
    EXPECT_TRUE(SymEngine::
                    eq(*assumptions.at(symbolic::symbol("j")).upper_bound(),
                       *symbolic::sub(symbolic::symbol("N"), symbolic::integer(1))));
    ASSERT_FALSE(assumptions.at(symbolic::symbol("i")).tight_lower_bound().is_null());
    EXPECT_TRUE(SymEngine::eq(*assumptions.at(symbolic::symbol("i")).tight_lower_bound(), *symbolic::integer(0)));
    ASSERT_FALSE(assumptions.at(symbolic::symbol("i")).tight_upper_bound().is_null());
    EXPECT_TRUE(SymEngine::
                    eq(*assumptions.at(symbolic::symbol("i")).tight_upper_bound(),
                       *symbolic::sub(symbolic::symbol("N"), symbolic::integer(2))));
    ASSERT_FALSE(assumptions.at(symbolic::symbol("j")).tight_lower_bound().is_null());
    EXPECT_TRUE(SymEngine::
                    eq(*assumptions.at(symbolic::symbol("j")).tight_lower_bound(),
                       *symbolic::add(indvar, symbolic::integer(1))));
    ASSERT_FALSE(assumptions.at(symbolic::symbol("j")).tight_upper_bound().is_null());
    EXPECT_TRUE(SymEngine::
                    eq(*assumptions.at(symbolic::symbol("j")).tight_upper_bound(),
                       *symbolic::sub(symbolic::symbol("N"), symbolic::integer(1))));
    EXPECT_TRUE(assumptions.at(symbolic::symbol("i")).constant());
    EXPECT_TRUE(assumptions.at(symbolic::symbol("j")).constant());
    EXPECT_TRUE(symbolic::
                    eq(assumptions.at(symbolic::symbol("N")).lower_bound(),
                       symbolic::max(symbolic::add(symbolic::symbol("i"), symbolic::integer(2)), symbolic::one())));
    EXPECT_TRUE(symbolic::eq(assumptions.at(symbolic::symbol("N")).upper_bound(), SymEngine::Inf));
    EXPECT_TRUE(assumptions.at(symbolic::symbol("N")).tight_lower_bound().is_null());
    EXPECT_TRUE(assumptions.at(symbolic::symbol("N")).tight_upper_bound().is_null());
    EXPECT_TRUE(assumptions.at(symbolic::symbol("N")).constant());
    EXPECT_EQ(analysis.parameters().size(), 1);
    EXPECT_TRUE(analysis.is_parameter("N"));
    EXPECT_FALSE(analysis.is_parameter("i"));
    EXPECT_FALSE(analysis.is_parameter("j"));
}

TEST(AssumptionsAnalysisTest, IndexAccess) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc_unsigned(types::PrimitiveType::UInt64);
    types::Scalar desc_signed(types::PrimitiveType::Int64);
    builder.add_container("N", desc_unsigned, true);
    builder.add_container("i", desc_signed);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("a", desc, true);
    builder.add_container("b", desc, true);
    builder.add_container("c", desc, true);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);

    // Perform FMA
    auto& block = builder.add_block(loop.root());
    auto& a = builder.add_access(block, "a");
    auto& b = builder.add_access(block, "b");
    auto& c1 = builder.add_access(block, "c");
    auto& c2 = builder.add_access(block, "c");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_fma, {"_out"}, {"_in1", "_in2", "_in3"});
    builder.add_computational_memlet(block, a, tasklet, "_in1", {symbolic::add(indvar, symbolic::one())});
    builder.add_computational_memlet(block, b, tasklet, "_in2", {symbolic::sub(indvar, symbolic::one())});
    builder.add_computational_memlet(block, c1, tasklet, "_in3", {indvar});
    builder.add_computational_memlet(block, tasklet, "_out", c2, {indvar});

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto& assumptions = analysis.get(loop.root());

    // Check
    EXPECT_EQ(assumptions.size(), 2);
    EXPECT_TRUE(symbolic::
                    eq(assumptions.at(symbolic::symbol("i")).lower_bound(),
                       SymEngine::max({symbolic::zero(), SymEngine::neg(symbolic::one()), symbolic::one()})));
    EXPECT_TRUE(symbolic::
                    eq(assumptions.at(symbolic::symbol("i")).upper_bound(),
                       SymEngine::min(
                           {symbolic::sub(symbolic::symbol("N"), symbolic::integer(1)),
                            symbolic::sub(symbolic::dynamic_sizeof(symbolic::symbol("a")), symbolic::integer(2)),
                            symbolic::dynamic_sizeof(symbolic::symbol("b")),
                            symbolic::sub(symbolic::dynamic_sizeof(symbolic::symbol("c")), symbolic::one())}
                       )));
    EXPECT_TRUE(symbolic::eq(assumptions.at(symbolic::symbol("i")).tight_lower_bound(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::
                    eq(assumptions.at(symbolic::symbol("i")).tight_upper_bound(),
                       symbolic::sub(symbolic::symbol("N"), symbolic::integer(1))));
    EXPECT_TRUE(assumptions.at(symbolic::symbol("i")).constant());
    EXPECT_TRUE(symbolic::eq(assumptions.at(symbolic::symbol("N")).lower_bound(), symbolic::one()));
    EXPECT_TRUE(symbolic::eq(assumptions.at(symbolic::symbol("N")).upper_bound(), SymEngine::Inf));
    EXPECT_TRUE(assumptions.at(symbolic::symbol("N")).tight_lower_bound().is_null());
    EXPECT_TRUE(assumptions.at(symbolic::symbol("N")).tight_upper_bound().is_null());
    EXPECT_TRUE(assumptions.at(symbolic::symbol("N")).constant());
    EXPECT_EQ(analysis.parameters().size(), 4);
    EXPECT_TRUE(analysis.is_parameter("N"));
    EXPECT_FALSE(analysis.is_parameter("i"));
    EXPECT_TRUE(analysis.is_parameter("a"));
    EXPECT_TRUE(analysis.is_parameter("b"));
    EXPECT_TRUE(analysis.is_parameter("c"));
}

TEST(AssumptionsAnalysisTest, ArrayBounds) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array desc(base_desc, symbolic::integer(400));
    types::Pointer desc2(desc);
    builder.add_container("A", desc2);

    // Define loop
    auto indvar = symbolic::symbol("i");
    auto bound = symbolic::symbol("N");
    auto condition = symbolic::Lt(indvar, bound);
    auto init = symbolic::zero();
    auto update = symbolic::add(indvar, symbolic::one());
    auto& loop = builder.add_for(root, indvar, condition, init, update);

    auto indvar_2 = symbolic::symbol("j");
    auto bound_2 = symbolic::symbol("M");
    auto condition_2 = symbolic::Lt(indvar_2, bound_2);
    auto init_2 = symbolic::zero();
    auto update_2 = symbolic::add(indvar_2, symbolic::one());
    auto& loop2 = builder.add_for(loop.root(), indvar_2, condition_2, init_2, update_2);

    // Add computation
    auto& block = builder.add_block(loop2.root());
    auto& two = builder.add_constant(block, "2", base_desc);
    auto& A1 = builder.add_access(block, "A");
    auto& A2 = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, {"_out"}, {"_in1", "_in2"});
    builder.add_computational_memlet(block, two, tasklet, "_in1", {});
    builder.add_computational_memlet(block, A1, tasklet, "_in2", {indvar, indvar_2});
    builder.add_computational_memlet(block, tasklet, "_out", A2, {indvar, indvar_2});

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto& assumptions = analysis.get(loop2.root());

    // Check
    EXPECT_EQ(assumptions.size(), 4);
    EXPECT_TRUE(symbolic::eq(assumptions.at(indvar).lower_bound(), symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(
        assumptions.at(indvar).upper_bound(),
        symbolic::min(symbolic::sub(symbolic::integer(400), symbolic::one()), symbolic::sub(bound, symbolic::one()))
    ));
    EXPECT_TRUE(symbolic::eq(assumptions.at(indvar).tight_lower_bound(), symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(assumptions.at(indvar).tight_upper_bound(), symbolic::sub(bound, symbolic::one())));
    EXPECT_TRUE(symbolic::eq(assumptions.at(indvar_2).lower_bound(), symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(assumptions.at(indvar_2).upper_bound(), symbolic::sub(bound_2, symbolic::one())));
    EXPECT_TRUE(symbolic::eq(assumptions.at(indvar_2).tight_lower_bound(), symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(assumptions.at(indvar_2).tight_upper_bound(), symbolic::sub(bound_2, symbolic::one())));
    EXPECT_TRUE(symbolic::eq(assumptions.at(bound).lower_bound(), symbolic::one()));
    EXPECT_TRUE(symbolic::eq(assumptions.at(bound).upper_bound(), SymEngine::Inf));
    EXPECT_TRUE(assumptions.at(bound).tight_lower_bound().is_null());
    EXPECT_TRUE(assumptions.at(bound).tight_upper_bound().is_null());
    EXPECT_TRUE(symbolic::eq(assumptions.at(bound_2).lower_bound(), symbolic::one()));
    EXPECT_TRUE(symbolic::eq(assumptions.at(bound_2).upper_bound(), SymEngine::Inf));
    EXPECT_TRUE(assumptions.at(bound_2).tight_lower_bound().is_null());
    EXPECT_TRUE(assumptions.at(bound_2).tight_upper_bound().is_null());
}
