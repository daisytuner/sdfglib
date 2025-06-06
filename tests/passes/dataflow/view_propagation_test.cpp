#include "sdfg/passes/dataflow/view_propagation.h"

#include <gtest/gtest.h>

#include <ostream>

#include "sdfg/builder/structured_sdfg_builder.h"

using namespace sdfg;

TEST(ViewPropagationTest, Simple) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Double);
    types::Pointer desc_ptr(desc);
    builder.add_container("A", desc_ptr, true);
    builder.add_container("a", desc_ptr);

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root);
    auto& viewed_node = builder.add_access(block1, "A");
    auto& viewing_node = builder.add_access(block1, "a");
    builder.add_memlet(block1, viewed_node, "void", viewing_node, "refs", {symbolic::integer(0)});

    auto& block2 = builder.add_block(root);
    auto& output_node = builder.add_access(block2, "a");
    auto& tasklet =
        builder.add_tasklet(block2, data_flow::TaskletCode::assign, {"_out", desc}, {{"0", desc}});
    builder.add_memlet(block2, tasklet, "_out", output_node, "void", {symbolic::integer(0)});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::ViewPropagation pass;
    pass.run(builder_opt, analysis_manager);
    sdfg = builder_opt.move();

    // Check result
    EXPECT_EQ(output_node.data(), "A");
}
