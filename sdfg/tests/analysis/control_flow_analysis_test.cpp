#include "sdfg/analysis/control_flow_analysis.h"

#include <gtest/gtest.h>

#include <gtest/gtest.h>

#include <unordered_map>
#include <unordered_set>

#include "sdfg/analysis/users.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/element.h"
#include "sdfg/symbolic/symbolic.h"

using namespace sdfg;

TEST(ControlFlowAnalysisTest, Sequence) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    builder.add_container("A", base_desc);

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root);
    auto& block2 = builder.add_block(root);

    // Run analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    auto& analysis = analysis_manager.get<analysis::ControlFlowAnalysis>();

    EXPECT_EQ(analysis.exits().size(), 1);
    EXPECT_NE(analysis.exits().find(&block2), analysis.exits().end());
    EXPECT_TRUE(analysis.dominates(block1, block2));
    EXPECT_FALSE(analysis.dominates(block2, block1));
    EXPECT_TRUE(analysis.post_dominates(block2, block1));
    EXPECT_FALSE(analysis.post_dominates(block1, block2));
}

TEST(ControlFlowAnalysisTest, IfElse) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    builder.add_container("A", base_desc);

    auto& root = builder.subject().root();

    auto& init_block = builder.add_block(root);
    auto& ifelse = builder.add_if_else(root);
    auto& then_branch = builder.add_case(ifelse, symbolic::__true__());
    auto& then_block = builder.add_block(then_branch);
    auto& else_branch = builder.add_case(ifelse, symbolic::__false__());
    auto& else_block = builder.add_block(else_branch);
    auto& join_block = builder.add_block(root);

    // Run analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    auto& analysis = analysis_manager.get<analysis::ControlFlowAnalysis>();

    EXPECT_EQ(analysis.exits().size(), 1);
    EXPECT_NE(analysis.exits().find(&join_block), analysis.exits().end());
    EXPECT_TRUE(analysis.dominates(init_block, join_block));
    EXPECT_TRUE(analysis.dominates(init_block, then_block));
    EXPECT_TRUE(analysis.dominates(init_block, else_block));

    EXPECT_FALSE(analysis.dominates(then_block, join_block));
    EXPECT_FALSE(analysis.dominates(else_block, join_block));
    EXPECT_FALSE(analysis.post_dominates(then_block, init_block));
    EXPECT_FALSE(analysis.post_dominates(else_block, init_block));

    EXPECT_TRUE(analysis.post_dominates(join_block, init_block));
    EXPECT_TRUE(analysis.post_dominates(join_block, then_block));
    EXPECT_TRUE(analysis.post_dominates(join_block, else_block));
}

TEST(ControlFlowAnalysisTest, While) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    builder.add_container("A", base_desc);

    auto& root = builder.subject().root();

    auto& init_block = builder.add_block(root);
    auto& while_node = builder.add_while(root);
    auto& body_block = builder.add_block(while_node.root());
    auto& exit_block = builder.add_block(root);

    // Run analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    auto& analysis = analysis_manager.get<analysis::ControlFlowAnalysis>();

    EXPECT_EQ(analysis.exits().size(), 1);
    EXPECT_NE(analysis.exits().find(&exit_block), analysis.exits().end());
    EXPECT_TRUE(analysis.dominates(init_block, exit_block));
    EXPECT_TRUE(analysis.dominates(init_block, body_block));

    EXPECT_FALSE(analysis.dominates(body_block, exit_block));
    EXPECT_FALSE(analysis.post_dominates(body_block, init_block));

    EXPECT_TRUE(analysis.post_dominates(exit_block, init_block));
    EXPECT_TRUE(analysis.post_dominates(exit_block, body_block));
}

TEST(ControlFlowAnalysisTest, StructuredLoop) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    builder.add_container("A", base_desc);
    builder.add_container("i", base_desc);

    auto& root = builder.subject().root();

    auto& init_block = builder.add_block(root);
    auto& for_node = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(1),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );
    auto& body_block = builder.add_block(for_node.root());
    auto& exit_block = builder.add_block(root);

    // Run analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    auto& analysis = analysis_manager.get<analysis::ControlFlowAnalysis>();

    EXPECT_EQ(analysis.exits().size(), 1);
    EXPECT_NE(analysis.exits().find(&exit_block), analysis.exits().end());
    EXPECT_TRUE(analysis.dominates(init_block, exit_block));
    EXPECT_TRUE(analysis.dominates(init_block, body_block));

    EXPECT_FALSE(analysis.dominates(body_block, exit_block));
    EXPECT_FALSE(analysis.post_dominates(body_block, init_block));

    EXPECT_TRUE(analysis.post_dominates(exit_block, init_block));
    EXPECT_TRUE(analysis.post_dominates(exit_block, body_block));
}
