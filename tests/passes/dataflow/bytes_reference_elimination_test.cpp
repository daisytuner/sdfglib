#include "sdfg/passes/dataflow/byte_reference_elimination.h"

#include <gtest/gtest.h>

#include <ostream>

#include "sdfg/builder/structured_sdfg_builder.h"

using namespace sdfg;

TEST(BytesReferenceEliminationTest, ReferenceToComputational) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc_double(types::PrimitiveType::Double);
    types::Scalar desc_int8(types::PrimitiveType::Int8);
    types::Pointer desc_ptr_double(desc_double);
    types::Pointer desc_ptr_int8(desc_int8);
    types::Pointer opaque_desc;

    builder.add_container("A", opaque_desc, true);
    builder.add_container("a", opaque_desc);

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root);
    auto& a_input = builder.add_access(block1, "A");
    auto& a_output = builder.add_access(block1, "a");
    builder.add_reference_memlet(block1, a_input, a_output, {symbolic::integer(16)}, desc_ptr_int8);

    auto& block2 = builder.add_block(root);
    auto& input_node = builder.add_access(block2, "a");
    auto& output_node = builder.add_access(block2, "a");
    auto& tasklet = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in0"});
    auto& iedge =
        builder.add_computational_memlet(block2, input_node, tasklet, "_in0", {symbolic::integer(1)}, desc_ptr_double);
    auto& oedge =
        builder.add_computational_memlet(block2, tasklet, "_out", output_node, {symbolic::integer(2)}, desc_ptr_double);

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::ByteReferenceElimination pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    // Check result
    EXPECT_EQ(input_node.data(), "A");
    EXPECT_EQ(output_node.data(), "A");

    EXPECT_EQ(iedge.subset().size(), 1);
    EXPECT_TRUE(symbolic::eq(iedge.subset()[0], symbolic::integer(3)));
    EXPECT_EQ(iedge.base_type(), desc_ptr_double);

    EXPECT_EQ(oedge.subset().size(), 1);
    EXPECT_TRUE(symbolic::eq(oedge.subset()[0], symbolic::integer(4)));
    EXPECT_EQ(oedge.base_type(), desc_ptr_double);
}

TEST(BytesReferenceEliminationTest, ReferenceToReference) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc_double(types::PrimitiveType::Double);
    types::Scalar desc_int8(types::PrimitiveType::Int8);
    types::Pointer desc_ptr_double(desc_double);
    types::Pointer desc_ptr_int8(desc_int8);
    types::Pointer opaque_desc;

    builder.add_container("A", opaque_desc, true);
    builder.add_container("a", opaque_desc);
    builder.add_container("b", opaque_desc);


    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root);
    auto& a_input = builder.add_access(block1, "A");
    auto& a_output = builder.add_access(block1, "a");
    builder.add_reference_memlet(block1, a_input, a_output, {symbolic::integer(16)}, desc_ptr_int8);

    auto& block2 = builder.add_block(root);
    auto& input_node = builder.add_access(block2, "a");
    auto& output_node = builder.add_access(block2, "b");
    auto& ref_edge =
        builder.add_reference_memlet(block2, input_node, output_node, {symbolic::integer(1)}, desc_ptr_double);

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::ByteReferenceElimination pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    // Check result
    EXPECT_EQ(input_node.data(), "A");
    EXPECT_EQ(output_node.data(), "b");

    EXPECT_EQ(ref_edge.subset().size(), 1);
    EXPECT_TRUE(symbolic::eq(ref_edge.subset()[0], symbolic::integer(3)));
    EXPECT_EQ(ref_edge.base_type(), desc_ptr_double);
}
