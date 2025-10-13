#include "sdfg/passes/structured_control_flow/dead_cfg_elimination.h"

#include <gtest/gtest.h>

#include <ostream>

#include "sdfg/builder/sdfg_builder.h"
#include "sdfg/builder/structured_sdfg_builder.h"

using namespace sdfg;

TEST(DeadCFGEliminationTest, VoidReturn) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& root = builder.subject().root();
    auto& return_node = builder.add_return(root, "");
    EXPECT_EQ(root.size(), 1);

    // Dead CFG Elimination
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::DeadCFGElimination dce_pass;
    EXPECT_TRUE(dce_pass.run(builder, analysis_manager));

    EXPECT_EQ(root.size(), 0);
}

TEST(DeadCFGEliminationTest, UndefReturn) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& root = builder.subject().root();
    auto& return_node = builder.add_constant_return(root, "", types::Scalar(types::PrimitiveType::Int32));
    EXPECT_EQ(root.size(), 1);

    // Dead CFG Elimination
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::DeadCFGElimination dce_pass;
    EXPECT_TRUE(dce_pass.run(builder, analysis_manager));

    EXPECT_EQ(root.size(), 0);
}

TEST(DeadCFGEliminationTest, AssignmentsAfterReturn) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type, true);

    auto& root = builder.subject().root();
    auto& return_node = builder.add_return(root, "i", {{sdfg::symbolic::symbol("i"), sdfg::symbolic::integer(10)}});
    EXPECT_EQ(root.size(), 1);
    EXPECT_EQ(root.at(0).second.size(), 1);

    // Dead CFG Elimination
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::DeadCFGElimination dce_pass;
    EXPECT_TRUE(dce_pass.run(builder, analysis_manager));

    EXPECT_EQ(root.size(), 1);
    EXPECT_EQ(root.at(0).second.size(), 0);
}

TEST(DeadCFGEliminationTest, NodesAfterReturn) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type, true);

    auto& root = builder.subject().root();
    auto& return_node = builder.add_return(root, "i", {{sdfg::symbolic::symbol("i"), sdfg::symbolic::integer(10)}});
    auto& block = builder.add_block(root);

    EXPECT_EQ(root.size(), 2);
    EXPECT_EQ(root.at(0).second.size(), 1);

    // Dead CFG Elimination
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::DeadCFGElimination dce_pass;
    EXPECT_TRUE(dce_pass.run(builder, analysis_manager));

    EXPECT_EQ(root.size(), 1);
    EXPECT_EQ(root.at(0).second.size(), 0);
}
