#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/tensor/conv_node.h"

using namespace sdfg;

// Test expansion of Conv2D with simple parameters
TEST(ConvNodeExpansionTest, Conv2D_SimpleExpansion) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    // Create simple 2D convolution: 1 batch, 1 input channel, 1 output channel
    // Input: [1, 1, 4, 4], Kernel: [1, 1, 3, 3], Output: [1, 1, 2, 2] with stride=1, no padding
    
    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    builder.add_container("input", desc_ptr);   // 1*1*4*4 = 16 floats
    builder.add_container("weights", desc_ptr); // 1*1*3*3 = 9 floats
    builder.add_container("output", desc_ptr);  // 1*1*2*2 = 4 floats

    auto& block = builder.add_block(sdfg.root());

    auto& input_node = builder.add_access(block, "input");
    auto& weights_node = builder.add_access(block, "weights");
    auto& output_node = builder.add_access(block, "output");

    std::vector<symbolic::Expression> kernel_shape = {symbolic::integer(3), symbolic::integer(3)};
    std::vector<symbolic::Expression> strides = {symbolic::integer(1), symbolic::integer(1)};
    std::vector<symbolic::Expression> pads = {symbolic::integer(0), symbolic::integer(0), 
                                               symbolic::integer(0), symbolic::integer(0)};
    std::vector<symbolic::Expression> dilations = {symbolic::integer(1), symbolic::integer(1)};
    auto group = symbolic::integer(1);

    auto& conv_node = static_cast<math::tensor::ConvNode&>(builder.add_library_node<math::tensor::ConvNode>(
        block, DebugInfo(), kernel_shape, strides, pads, dilations, group
    ));

    builder.add_computational_memlet(block, input_node, conv_node, "X", {}, desc_ptr, block.debug_info());
    builder.add_computational_memlet(block, weights_node, conv_node, "W", {}, desc_ptr, block.debug_info());
    builder.add_computational_memlet(block, conv_node, "Y", output_node, {}, desc_ptr, block.debug_info());

    // Validate the SDFG before expansion
    EXPECT_NO_THROW(sdfg.validate());

    // Try to expand the node - expansion should now succeed for 2D convolution
    analysis::AnalysisManager analysis_manager(sdfg);
    bool expanded = conv_node.expand(builder, analysis_manager);
    
    // Expansion should now be implemented and return true
    EXPECT_TRUE(expanded);
}

// Test that expansion is not attempted with unsupported parameters
TEST(ConvNodeExpansionTest, Conv2D_ExpansionNotImplemented) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    builder.add_container("input", desc_ptr);
    builder.add_container("weights", desc_ptr);
    builder.add_container("output", desc_ptr);

    auto& block = builder.add_block(sdfg.root());

    auto& input_node = builder.add_access(block, "input");
    auto& weights_node = builder.add_access(block, "weights");
    auto& output_node = builder.add_access(block, "output");

    std::vector<symbolic::Expression> kernel_shape = {symbolic::integer(3), symbolic::integer(3)};
    std::vector<symbolic::Expression> strides = {symbolic::integer(2), symbolic::integer(2)};
    std::vector<symbolic::Expression> pads = {symbolic::integer(1), symbolic::integer(1), 
                                               symbolic::integer(1), symbolic::integer(1)};
    std::vector<symbolic::Expression> dilations = {symbolic::integer(1), symbolic::integer(1)};
    auto group = symbolic::integer(1);

    auto& conv_node = static_cast<math::tensor::ConvNode&>(builder.add_library_node<math::tensor::ConvNode>(
        block, DebugInfo(), kernel_shape, strides, pads, dilations, group
    ));

    builder.add_computational_memlet(block, input_node, conv_node, "X", {}, desc_ptr, block.debug_info());
    builder.add_computational_memlet(block, weights_node, conv_node, "W", {}, desc_ptr, block.debug_info());
    builder.add_computational_memlet(block, conv_node, "Y", output_node, {}, desc_ptr, block.debug_info());

    analysis::AnalysisManager analysis_manager(sdfg);
    bool expanded = conv_node.expand(builder, analysis_manager);
    
    // Expansion should succeed for 2D convolutions with stride > 1
    EXPECT_TRUE(expanded);
}

// Test 1D convolution expansion
TEST(ConvNodeExpansionTest, Conv1D_Expansion) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    builder.add_container("input", desc_ptr);
    builder.add_container("weights", desc_ptr);
    builder.add_container("output", desc_ptr);

    auto& block = builder.add_block(sdfg.root());

    auto& input_node = builder.add_access(block, "input");
    auto& weights_node = builder.add_access(block, "weights");
    auto& output_node = builder.add_access(block, "output");

    std::vector<symbolic::Expression> kernel_shape = {symbolic::integer(5)};
    std::vector<symbolic::Expression> strides = {symbolic::integer(1)};
    std::vector<symbolic::Expression> pads = {symbolic::integer(2), symbolic::integer(2)};
    std::vector<symbolic::Expression> dilations = {symbolic::integer(1)};
    auto group = symbolic::integer(1);

    auto& conv_node = static_cast<math::tensor::ConvNode&>(builder.add_library_node<math::tensor::ConvNode>(
        block, DebugInfo(), kernel_shape, strides, pads, dilations, group
    ));

    builder.add_computational_memlet(block, input_node, conv_node, "X", {}, desc_ptr, block.debug_info());
    builder.add_computational_memlet(block, weights_node, conv_node, "W", {}, desc_ptr, block.debug_info());
    builder.add_computational_memlet(block, conv_node, "Y", output_node, {}, desc_ptr, block.debug_info());

    // Validate the SDFG before expansion
    EXPECT_NO_THROW(sdfg.validate());

    analysis::AnalysisManager analysis_manager(sdfg);
    bool expanded = conv_node.expand(builder, analysis_manager);
    
    // Expansion should succeed for 1D convolutions
    EXPECT_TRUE(expanded);
}

// Test 3D convolution expansion
TEST(ConvNodeExpansionTest, Conv3D_Expansion) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    builder.add_container("input", desc_ptr);
    builder.add_container("weights", desc_ptr);
    builder.add_container("output", desc_ptr);

    auto& block = builder.add_block(sdfg.root());

    auto& input_node = builder.add_access(block, "input");
    auto& weights_node = builder.add_access(block, "weights");
    auto& output_node = builder.add_access(block, "output");

    std::vector<symbolic::Expression> kernel_shape = {symbolic::integer(3), symbolic::integer(3), symbolic::integer(3)};
    std::vector<symbolic::Expression> strides = {symbolic::integer(1), symbolic::integer(1), symbolic::integer(1)};
    std::vector<symbolic::Expression> pads = {symbolic::integer(1), symbolic::integer(1), symbolic::integer(1),
                                               symbolic::integer(1), symbolic::integer(1), symbolic::integer(1)};
    std::vector<symbolic::Expression> dilations = {symbolic::integer(1), symbolic::integer(1), symbolic::integer(1)};
    auto group = symbolic::integer(1);

    auto& conv_node = static_cast<math::tensor::ConvNode&>(builder.add_library_node<math::tensor::ConvNode>(
        block, DebugInfo(), kernel_shape, strides, pads, dilations, group
    ));

    builder.add_computational_memlet(block, input_node, conv_node, "X", {}, desc_ptr, block.debug_info());
    builder.add_computational_memlet(block, weights_node, conv_node, "W", {}, desc_ptr, block.debug_info());
    builder.add_computational_memlet(block, conv_node, "Y", output_node, {}, desc_ptr, block.debug_info());

    // Validate the SDFG before expansion
    EXPECT_NO_THROW(sdfg.validate());

    analysis::AnalysisManager analysis_manager(sdfg);
    bool expanded = conv_node.expand(builder, analysis_manager);
    
    // Expansion should succeed for 3D convolutions
    EXPECT_TRUE(expanded);
}
