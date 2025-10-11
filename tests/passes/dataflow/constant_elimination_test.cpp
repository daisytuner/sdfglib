#include <gtest/gtest.h>

#include <ostream>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/passes/dataflow/constant_elimination.h"

using namespace sdfg;

TEST(ConstantEliminationTest, Transition) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    builder.add_container("j", desc, true);

    builder.add_block(builder.subject().root(), {{symbolic::symbol("i"), symbolic::symbol("j")}});
    builder.add_block(builder.subject().root(), {{symbolic::symbol("i"), symbolic::symbol("j")}});

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::ConstantElimination pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    // Check result
    auto& sdfg = builder.subject();

    auto& trans1 = sdfg.root().at(0).second;
    EXPECT_EQ(trans1.assignments().size(), 1);
    EXPECT_TRUE(symbolic::eq(trans1.assignments().at(symbolic::symbol("i")), symbolic::symbol("j")));

    auto& trans2 = sdfg.root().at(1).second;
    EXPECT_EQ(trans2.assignments().size(), 0);
}

TEST(ConstantEliminationTest, Transition_ptr) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Bool);
    types::Pointer ptr_desc;
    builder.add_container("i", desc);
    builder.add_container("p", ptr_desc, true);

    builder.add_block(builder.subject().root(), {{symbolic::symbol("i"), symbolic::Eq(symbolic::symbol("p"), symbolic::__nullptr__())}});
    builder.add_block(builder.subject().root(), {{symbolic::symbol("i"), symbolic::Eq(symbolic::symbol("p"), symbolic::__nullptr__())}});

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::ConstantElimination pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    // Check result
    auto& sdfg = builder.subject();

    auto& trans1 = sdfg.root().at(0).second;
    EXPECT_EQ(trans1.assignments().size(), 1);
    EXPECT_TRUE(symbolic::eq(trans1.assignments().at(symbolic::symbol("i")), symbolic::Eq(symbolic::symbol("p"), symbolic::__nullptr__())));

    auto& trans2 = sdfg.root().at(1).second;
    EXPECT_EQ(trans2.assignments().size(), 0);
}
