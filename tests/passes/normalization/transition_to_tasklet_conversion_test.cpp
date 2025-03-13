#include "sdfg/passes/normalization/transition_to_tasklet_conversion.h"

#include <gtest/gtest.h>

using namespace sdfg;

TEST(TransitionToTaskletConversionTest, Basic) {
    builder::StructuredSDFGBuilder builder("sdfg");

    types::Scalar sym_desc(types::PrimitiveType::UInt32);
    builder.add_container("a", sym_desc, true);
    builder.add_container("i", sym_desc);
    auto sym = symbolic::symbol("i");
    auto sym2 = symbolic::symbol("a");

    auto& root = builder.subject().root();

    auto& block = builder.add_block(root, {{sym2, sym}});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::TransitionToTaskletConversionPass pass;
    EXPECT_TRUE(pass.run(builder_opt, analysis_manager));

    sdfg = builder_opt.move();
    auto& root_opt = sdfg->root();
    EXPECT_EQ(root_opt.size(), 2);

    auto block_opt1 = root_opt.at(0);
    EXPECT_EQ(block_opt1.second.size(), 0);
    auto block_opt2 = root_opt.at(1);
    EXPECT_EQ(block_opt2.second.size(), 0);

    auto& dataflow1 = dynamic_cast<structured_control_flow::Block&>(block_opt1.first);
    EXPECT_EQ(dataflow1.dataflow().nodes().size(), 0);
    auto& dataflow2 = dynamic_cast<structured_control_flow::Block&>(block_opt2.first);
    EXPECT_EQ(dataflow2.dataflow().nodes().size(), 5);
    EXPECT_EQ(dataflow2.dataflow().tasklets().size(), 2);
    EXPECT_EQ(dataflow2.dataflow().data_nodes().size(), 3);
}
