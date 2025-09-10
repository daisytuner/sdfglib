#include "sdfg/passes/debug_info_propagation.h"

#include <gtest/gtest.h>

#include "sdfg/debug_info.h"
#include "sdfg/element.h"

using namespace sdfg;

TEST(DebugTablePropagationTest, BlockPropagation_Node) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU, DebugTable());

    DebugLoc debug_loc("main.c", "main", 1, 1, true);
    DebugInfo debug_info_element(debug_loc);

    size_t index = builder.add_debug_info_element(debug_info_element);

    DebugInfoRegion debug_info_region({index}, builder.debug_info().instructions());

    types::Scalar desc_element(types::PrimitiveType::Double);
    types::Array desc_array(desc_element, symbolic::integer(10));
    builder.add_container("A", desc_array);

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root);

    auto& node1_1 = builder.add_access(block1, "A");
    auto& node2_1 = builder.add_access(block1, "A");
    auto& tasklet_1 =
        builder.add_tasklet(block1, data_flow::TaskletCode::fma, "_out", {"2", "_in", "1"}, debug_info_region);
    builder.add_computational_memlet(block1, node1_1, tasklet_1, "_in", {symbolic::integer(0)});
    builder.add_computational_memlet(block1, tasklet_1, "_out", node2_1, {symbolic::integer(0)});

    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::DebugTablePropagation debug_info_propagation_pass;
    debug_info_propagation_pass.run(builder, analysis_manager);

    EXPECT_TRUE(builder.subject().root().debug_info().has());
    EXPECT_EQ(builder.subject().root().debug_info().filename(), "main.c");
    EXPECT_EQ(builder.subject().root().debug_info().function(), "main");
    EXPECT_EQ(builder.subject().root().debug_info().start_line(), 1);
    EXPECT_EQ(builder.subject().root().debug_info().start_column(), 1);
    EXPECT_EQ(builder.subject().root().debug_info().end_line(), 1);
    EXPECT_EQ(builder.subject().root().debug_info().end_column(), 1);
}

TEST(DebugTablePropagationTest, BlockPropagation_Edge) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU, DebugTable());

    DebugLoc debug_loc1("main.c", "main", 1, 1, true);
    DebugInfo debug_info_element1(debug_loc1);
    size_t index1 = builder.add_debug_info_element(debug_info_element1);

    DebugLoc debug_loc2("main.c", "main", 2, 2, true);
    DebugInfo debug_info_element2(debug_loc2);
    size_t index2 = builder.add_debug_info_element(debug_info_element2);

    DebugInfoRegion debug_info1({index1}, builder.debug_info().instructions());
    DebugInfoRegion debug_info2({index2}, builder.debug_info().instructions());

    types::Scalar desc_element(types::PrimitiveType::Double);
    types::Array desc_array(desc_element, symbolic::integer(10));
    builder.add_container("A", desc_array);

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root);

    auto& node1_1 = builder.add_access(block1, "A");
    auto& node2_1 = builder.add_access(block1, "A");
    auto& tasklet_1 = builder.add_tasklet(block1, data_flow::TaskletCode::fma, "_out", {"2", "_in", "1"});
    builder.add_computational_memlet(block1, node1_1, tasklet_1, "_in", {symbolic::integer(0)}, debug_info1);
    builder.add_computational_memlet(block1, tasklet_1, "_out", node2_1, {symbolic::integer(0)}, debug_info2);

    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::DebugTablePropagation debug_info_propagation_pass;
    debug_info_propagation_pass.run(builder, analysis_manager);

    EXPECT_TRUE(builder.subject().root().debug_info().has());
    EXPECT_EQ(builder.subject().root().debug_info().filename(), "main.c");
    EXPECT_EQ(builder.subject().root().debug_info().start_line(), 1);
    EXPECT_EQ(builder.subject().root().debug_info().start_column(), 1);
    EXPECT_EQ(builder.subject().root().debug_info().end_line(), 2);
    EXPECT_EQ(builder.subject().root().debug_info().end_column(), 2);
}
