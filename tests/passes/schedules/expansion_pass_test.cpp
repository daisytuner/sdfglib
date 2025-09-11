#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/math.h"
#include "sdfg/passes/schedules/expansion_pass.h"

using namespace sdfg;

TEST(ExpansionPassTest, Simple) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Double);
    types::Array array_desc(desc, symbolic::integer(10));
    types::Array array_desc_2(array_desc, symbolic::integer(20));

    builder.add_container("input", array_desc_2);
    builder.add_container("output", array_desc_2);

    auto& block = builder.add_block(sdfg.root());

    auto& input_node = builder.add_access(block, "input");
    auto& output_node = builder.add_access(block, "output");
    auto& relu_node =
        static_cast<math::ml::ReLUNode&>(builder.add_library_node<math::ml::ReLUNode>(block, DebugInfoRegion()));

    builder.add_computational_memlet(
        block,
        input_node,
        relu_node,
        "X",
        {symbolic::integer(0), symbolic::integer(0)},
        {symbolic::integer(10), symbolic::integer(20)},
        array_desc_2,
        builder.debug_info().get_region(block.debug_info().indices())
    );
    builder.add_computational_memlet(
        block,
        relu_node,
        "Y",
        output_node,
        {symbolic::integer(0), symbolic::integer(0)},
        {symbolic::integer(10), symbolic::integer(20)},
        array_desc_2,
        builder.debug_info().get_region(block.debug_info().indices())
    );

    analysis::AnalysisManager analysis_manager(sdfg);
    passes::ExpansionPass expansion_pass;
    EXPECT_TRUE(expansion_pass.run(builder, analysis_manager));

    EXPECT_EQ(sdfg.root().size(), 1);
    auto new_sequence = dynamic_cast<structured_control_flow::Sequence*>(&sdfg.root().at(0).first);
    EXPECT_NE(new_sequence, nullptr);

    auto map_1 = dynamic_cast<structured_control_flow::Map*>(&new_sequence->at(0).first);
    EXPECT_NE(map_1, nullptr);
    EXPECT_EQ(map_1->root().size(), 1);

    auto map_2 = dynamic_cast<structured_control_flow::Map*>(&map_1->root().at(0).first);
    EXPECT_NE(map_2, nullptr);
    EXPECT_EQ(map_2->root().size(), 1);

    auto block_1 = dynamic_cast<structured_control_flow::Block*>(&map_2->root().at(0).first);
    EXPECT_NE(block_1, nullptr);
    EXPECT_EQ(block_1->dataflow().nodes().size(), 3);

    auto tasklet = *block_1->dataflow().tasklets().begin();
    EXPECT_EQ(tasklet->code(), data_flow::TaskletCode::max);
    EXPECT_EQ(tasklet->inputs().size(), 2);
    EXPECT_EQ(tasklet->inputs().at(0), "0.0f");
    EXPECT_EQ(tasklet->inputs().at(1), "_in");
    EXPECT_EQ(tasklet->output(), "_out");
}
