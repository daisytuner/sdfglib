#include "sdfg/passes/dataflow/reference_propagation.h"

#include <gtest/gtest.h>

#include <ostream>

#include "sdfg/builder/structured_sdfg_builder.h"

using namespace sdfg;

TEST(ReferencePropagationTest, ReferenceToComputational) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Double);
    types::Pointer desc_ptr(desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("a", opaque_desc);

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root);
    auto& a_input = builder.add_access(block1, "A");
    auto& a_output = builder.add_access(block1, "a");
    builder.add_reference_memlet(block1, a_input, a_output, {symbolic::integer(0)}, desc_ptr);

    auto& block2 = builder.add_block(root);
    auto& input_node = builder.add_access(block2, "a");
    auto& output_node = builder.add_access(block2, "a");
    auto& tasklet = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in0"});
    auto& iedge =
        builder.add_computational_memlet(block2, input_node, tasklet, "_in0", {symbolic::integer(0)}, desc_ptr);
    auto& oedge =
        builder.add_computational_memlet(block2, tasklet, "_out", output_node, {symbolic::integer(0)}, desc_ptr);

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

    EXPECT_EQ(iedge.subset().size(), 1);
    EXPECT_TRUE(symbolic::eq(iedge.subset()[0], symbolic::integer(0)));
    EXPECT_EQ(iedge.base_type(), desc_ptr);

    EXPECT_EQ(oedge.subset().size(), 1);
    EXPECT_TRUE(symbolic::eq(oedge.subset()[0], symbolic::integer(0)));
    EXPECT_EQ(oedge.base_type(), desc_ptr);
}

TEST(ReferencePropagationTest, ReferenceToComputational_Subsets) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc_base(types::PrimitiveType::Double);
    types::Pointer desc_base_ptr(desc_base);
    types::Array desc_array(desc_base, symbolic::integer(10));
    types::Pointer desc_array_ptr(desc_array);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("a", opaque_desc);

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root);
    auto& a_input = builder.add_access(block1, "A");
    auto& a_output = builder.add_access(block1, "a");
    builder
        .add_reference_memlet(block1, a_input, a_output, {symbolic::integer(1), symbolic::integer(2)}, desc_array_ptr);

    auto& block2 = builder.add_block(root);
    auto& input_node = builder.add_access(block2, "a");
    auto& output_node = builder.add_access(block2, "a");
    auto& tasklet = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in0"});
    auto& iedge =
        builder.add_computational_memlet(block2, input_node, tasklet, "_in0", {symbolic::integer(0)}, desc_base_ptr);
    auto& oedge =
        builder.add_computational_memlet(block2, tasklet, "_out", output_node, {symbolic::integer(0)}, desc_base_ptr);

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

    EXPECT_EQ(iedge.subset().size(), 2);
    EXPECT_TRUE(symbolic::eq(iedge.subset()[0], symbolic::integer(1)));
    EXPECT_TRUE(symbolic::eq(iedge.subset()[1], symbolic::integer(2)));

    EXPECT_EQ(oedge.subset().size(), 2);
    EXPECT_TRUE(symbolic::eq(oedge.subset()[0], symbolic::integer(1)));
    EXPECT_TRUE(symbolic::eq(oedge.subset()[1], symbolic::integer(2)));
}

TEST(ReferencePropagationTest, ReferenceToDereference_Src) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Pointer opaque_desc;
    types::Pointer ptr_desc(static_cast<const types::IType&>(opaque_desc));

    builder.add_container("A", opaque_desc, true);
    builder.add_container("A_", opaque_desc);
    builder.add_container("a", opaque_desc);

    auto& root = builder.subject().root();

    // Alias A -> A_
    {
        auto& block1 = builder.add_block(root);
        auto& a_input = builder.add_access(block1, "A");
        auto& a_output = builder.add_access(block1, "A_");

        types::Pointer opaque_desc;
        types::Pointer ptr_desc(static_cast<const types::IType&>(opaque_desc));
        builder.add_reference_memlet(block1, a_input, a_output, {symbolic::integer(0)}, ptr_desc);
    }

    // Load a = *A_
    auto& block2 = builder.add_block(root);
    auto& a_input = builder.add_access(block2, "A_");
    auto& a_output = builder.add_access(block2, "a");
    builder.add_dereference_memlet(block2, a_input, a_output, true, ptr_desc);

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::ReferencePropagation pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    // Check result
    EXPECT_EQ(a_input.data(), "A");
}

TEST(ReferencePropagationTest, ReferenceToDereference_Dst) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Pointer opaque_desc;
    types::Pointer ptr_desc(static_cast<const types::IType&>(opaque_desc));

    builder.add_container("A", opaque_desc, true);
    builder.add_container("A_", opaque_desc);
    builder.add_container("a", opaque_desc);

    auto& root = builder.subject().root();

    // Alias A -> A_
    {
        auto& block1 = builder.add_block(root);
        auto& a_input = builder.add_access(block1, "A");
        auto& a_output = builder.add_access(block1, "A_");

        types::Pointer opaque_desc;
        types::Pointer ptr_desc(static_cast<const types::IType&>(opaque_desc));
        builder.add_reference_memlet(block1, a_input, a_output, {symbolic::integer(0)}, ptr_desc);
    }

    // Load a = *A_
    auto& block2 = builder.add_block(root);
    auto& a_input = builder.add_access(block2, "a");
    auto& a_output = builder.add_access(block2, "A_");
    builder.add_dereference_memlet(block2, a_input, a_output, false, ptr_desc);

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::ReferencePropagation pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    // Check result
    EXPECT_EQ(a_output.data(), "A");
}

TEST(ReferencePropagationTest, ReferenceToDereference_NonTrivialSubset) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Pointer opaque_desc;
    types::Pointer ptr_desc(static_cast<const types::IType&>(opaque_desc));

    builder.add_container("A", opaque_desc, true);
    builder.add_container("A_", opaque_desc);
    builder.add_container("a", opaque_desc);

    auto& root = builder.subject().root();

    // Alias A -> A_
    {
        auto& block1 = builder.add_block(root);
        auto& a_input = builder.add_access(block1, "A");
        auto& a_output = builder.add_access(block1, "A_");

        types::Pointer opaque_desc;
        types::Pointer ptr_desc(static_cast<const types::IType&>(opaque_desc));
        builder.add_reference_memlet(block1, a_input, a_output, {symbolic::integer(1)}, ptr_desc);
    }

    // Load a = *A_
    auto& block2 = builder.add_block(root);
    auto& a_input = builder.add_access(block2, "a");
    auto& a_output = builder.add_access(block2, "A_");
    builder.add_dereference_memlet(block2, a_input, a_output, false, ptr_desc);

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::ReferencePropagation pass;
    EXPECT_FALSE(pass.run(builder, analysis_manager));

    // Check result
    EXPECT_EQ(a_output.data(), "A_");
}

TEST(ReferencePropagationTest, ReferenceToReference) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Pointer opaque_desc;
    types::Pointer ptr_desc(static_cast<const types::IType&>(opaque_desc));

    builder.add_container("A", opaque_desc, true);
    builder.add_container("B", opaque_desc, true);
    builder.add_container("a", opaque_desc);

    auto& root = builder.subject().root();

    // Alias A -> a
    {
        auto& block1 = builder.add_block(root);
        auto& a_input = builder.add_access(block1, "A");
        auto& a_output = builder.add_access(block1, "a");

        types::Pointer opaque_desc;
        types::Pointer ptr_desc(static_cast<const types::IType&>(opaque_desc));
        builder.add_reference_memlet(block1, a_input, a_output, {symbolic::integer(0)}, ptr_desc);
    }

    // Alias a+1 -> B
    auto& block2 = builder.add_block(root);
    auto& a_input = builder.add_access(block2, "a");
    auto& a_output = builder.add_access(block2, "B");
    builder.add_reference_memlet(block2, a_input, a_output, {symbolic::integer(1)}, ptr_desc);

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::ReferencePropagation pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    // Check result
    EXPECT_EQ(a_input.data(), "A");
}

TEST(ReferencePropagationTest, ReferenceToReference_SameContainer) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Pointer opaque_desc;
    types::Pointer ptr_desc(static_cast<const types::IType&>(opaque_desc));

    builder.add_container("A", opaque_desc, true);
    builder.add_container("B", opaque_desc, true);
    builder.add_container("a", opaque_desc);

    auto& root = builder.subject().root();

    // Alias A -> a
    {
        auto& block1 = builder.add_block(root);
        auto& a_input = builder.add_access(block1, "A");
        auto& a_output = builder.add_access(block1, "a");

        types::Pointer opaque_desc;
        types::Pointer ptr_desc(static_cast<const types::IType&>(opaque_desc));
        builder.add_reference_memlet(block1, a_input, a_output, {symbolic::integer(1)}, ptr_desc);
    }

    // Alias B -> a
    auto& block2 = builder.add_block(root);
    auto& a_input = builder.add_access(block2, "B");
    auto& a_output = builder.add_access(block2, "a");
    builder.add_reference_memlet(block2, a_input, a_output, {symbolic::integer(2)}, ptr_desc);

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::ReferencePropagation pass;
    EXPECT_FALSE(pass.run(builder, analysis_manager));

    // Check result
    EXPECT_EQ(a_output.data(), "a");
}

TEST(ReferencePropagationTest, StoreIntoSelf) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Pointer opaque_desc;
    types::Pointer ptr_desc(static_cast<const types::IType&>(opaque_desc));

    builder.add_container("A", opaque_desc, true);
    builder.add_container("a", opaque_desc);

    auto& root = builder.subject().root();

    // a = A + 1
    {
        auto& block1 = builder.add_block(root);
        auto& a_input = builder.add_access(block1, "A");
        auto& a_output = builder.add_access(block1, "a");

        types::Pointer opaque_desc;
        types::Pointer ptr_desc(static_cast<const types::IType&>(opaque_desc));
        builder.add_reference_memlet(block1, a_input, a_output, {symbolic::integer(1)}, ptr_desc);
    }

    // *a = a
    auto& block2 = builder.add_block(root);
    auto& a_input = builder.add_access(block2, "A");
    auto& a_output = builder.add_access(block2, "a");
    builder.add_dereference_memlet(block2, a_input, a_output, false, ptr_desc);

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::ReferencePropagation pass;
    EXPECT_FALSE(pass.run(builder, analysis_manager));
}

TEST(ReferencePropagationTest, PointerIteration) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Pointer opaque_desc;
    types::Pointer ptr_desc(static_cast<const types::IType&>(opaque_desc));

    builder.add_container("A", opaque_desc, true);
    builder.add_container("tmp", opaque_desc);

    auto& root = builder.subject().root();

    // a = A + 1
    {
        auto& block1 = builder.add_block(root);
        auto& a_input = builder.add_access(block1, "A");
        auto& a_output = builder.add_access(block1, "tmp");

        types::Pointer opaque_desc;
        types::Pointer ptr_desc(static_cast<const types::IType&>(opaque_desc));
        builder.add_reference_memlet(block1, a_input, a_output, {symbolic::one()}, ptr_desc);
    }

    // A = a
    auto& block2 = builder.add_block(root);
    auto& a_input = builder.add_access(block2, "tmp");
    auto& a_output = builder.add_access(block2, "A");
    builder.add_reference_memlet(block2, a_input, a_output, {symbolic::zero()}, ptr_desc);

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::ReferencePropagation pass;
    EXPECT_FALSE(pass.run(builder, analysis_manager));
}

TEST(ReferencePropagationTest, PointerIteration2) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Pointer opaque_desc;
    types::Pointer ptr_desc(static_cast<const types::IType&>(opaque_desc));

    builder.add_container("A", opaque_desc, true);
    builder.add_container("tmp", opaque_desc);

    auto& root = builder.subject().root();

    // a = A + 1
    {
        auto& block1 = builder.add_block(root);
        auto& a_input = builder.add_access(block1, "A");
        auto& a_output = builder.add_access(block1, "tmp");

        types::Pointer opaque_desc;
        types::Pointer ptr_desc(static_cast<const types::IType&>(opaque_desc));
        builder.add_reference_memlet(block1, a_input, a_output, {symbolic::zero()}, ptr_desc);
    }

    // A = a
    auto& block2 = builder.add_block(root);
    auto& a_input = builder.add_access(block2, "tmp");
    auto& a_output = builder.add_access(block2, "A");
    auto& ref_edge = builder.add_reference_memlet(block2, a_input, a_output, {symbolic::one()}, ptr_desc);

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::ReferencePropagation pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    EXPECT_EQ(a_input.data(), "A");
}
