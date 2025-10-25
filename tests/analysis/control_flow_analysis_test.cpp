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
