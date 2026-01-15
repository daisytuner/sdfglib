#include "sdfg/passes/dataflow/trivial_reference_conversion.h"

#include <gtest/gtest.h>

#include <ostream>

#include "sdfg/builder/structured_sdfg_builder.h"

using namespace sdfg;


TEST(TrivialReferenceConversion, PointerPointerToInt8) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("a", opaque_desc);

    auto& root = builder.subject().root();

    // A = &((void**)a)[0]
    auto& block = builder.add_block(root);
    auto& input_node = builder.add_access(block, "a");
    auto& output_node = builder.add_access(block, "A");
    auto& ref_edge = builder.add_reference_memlet(
        block, input_node, output_node, {symbolic::integer(0)}, types::Pointer(static_cast<types::IType&>(opaque_desc))
    );


    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::TrivialReferenceConversionPass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    types::Scalar desc_int8(types::PrimitiveType::Int8);
    types::Pointer desc_ptr_int8(desc_int8);

    // Check result
    EXPECT_EQ(input_node.data(), "a");
    EXPECT_EQ(output_node.data(), "A");
    EXPECT_EQ(ref_edge.subset().size(), 1);
    EXPECT_TRUE(symbolic::eq(ref_edge.subset()[0], symbolic::integer(0)));
    EXPECT_EQ(ref_edge.base_type(), desc_ptr_int8);
}
