#include "sdfg/passes/dataflow/reference_propagation.h"

#include <gtest/gtest.h>

#include <ostream>

#include "sdfg/builder/structured_sdfg_builder.h"

using namespace sdfg;

TEST(ReferencePropagationTest, ReferenceMemlet) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Double);
    types::Pointer desc_ptr(desc);
    builder.add_container("A", desc_ptr, true);
    builder.add_container("a", desc_ptr);

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root);
    auto& a_input = builder.add_access(block1, "A");
    auto& a_output = builder.add_access(block1, "a");
    builder.add_memlet(block1, a_input, "void", a_output, "ref", {symbolic::integer(0)});

    auto& block2 = builder.add_block(root);
    auto& input_node = builder.add_access(block2, "a");
    auto& output_node = builder.add_access(block2, "a");
    auto& tasklet =
        builder.add_tasklet(block2, data_flow::TaskletCode::add, {"_out", desc}, {{"_in0", desc}, {"1", desc}});
    builder.add_memlet(block2, input_node, "void", tasklet, "_in0", {symbolic::integer(0)});
    builder.add_memlet(block2, tasklet, "_out", output_node, "void", {symbolic::integer(0)});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::ReferencePropagation pass;
    EXPECT_TRUE(pass.run(builder_opt, analysis_manager));
    sdfg = builder_opt.move();

    // Check result
    EXPECT_EQ(input_node.data(), "A");
    EXPECT_EQ(output_node.data(), "A");
}

TEST(ReferencePropagationTest, DereferenceMemlet_Load) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Double);
    types::Pointer desc_ptr(desc);
    types::Pointer desc_ptr_2(static_cast<const types::IType&>(desc_ptr));
    builder.add_container("A", desc_ptr_2, true);
    builder.add_container("a", desc_ptr);

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root);
    auto& a_input = builder.add_access(block1, "A");
    auto& a_output = builder.add_access(block1, "a");
    builder.add_memlet(block1, a_input, "void", a_output, "deref", {symbolic::integer(0)});

    auto& block2 = builder.add_block(root);
    auto& output_node = builder.add_access(block2, "a");
    auto& tasklet = builder.add_tasklet(block2, data_flow::TaskletCode::assign, {"_out", desc}, {{"0", desc}});
    builder.add_memlet(block2, tasklet, "_out", output_node, "void", {symbolic::integer(0)});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::ReferencePropagation pass;
    EXPECT_FALSE(pass.run(builder_opt, analysis_manager));

    sdfg = builder_opt.move();

    // Check result
    EXPECT_EQ(output_node.data(), "a");
}

TEST(ReferencePropagationTest, DereferenceMemlet_Store) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Double);
    types::Pointer desc_ptr(desc);
    types::Pointer desc_ptr_2(static_cast<const types::IType&>(desc_ptr));
    builder.add_container("A", desc_ptr_2, true);
    builder.add_container("a", desc_ptr);

    auto& root = builder.subject().root();

    auto& block1 = builder.add_block(root);
    auto& output_node = builder.add_access(block1, "a");
    auto& tasklet = builder.add_tasklet(block1, data_flow::TaskletCode::assign, {"_out", desc}, {{"0", desc}});
    builder.add_memlet(block1, tasklet, "_out", output_node, "void", {symbolic::integer(0)});

    auto& block2 = builder.add_block(root);
    auto& a_input = builder.add_access(block2, "a");
    auto& a_output = builder.add_access(block2, "A");
    builder.add_memlet(block2, a_input, "deref", a_output, "void", {symbolic::integer(0)});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::ReferencePropagation pass;
    EXPECT_FALSE(pass.run(builder_opt, analysis_manager));

    sdfg = builder_opt.move();

    // Check result
    EXPECT_EQ(output_node.data(), "a");
}
