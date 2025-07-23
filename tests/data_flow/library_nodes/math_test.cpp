#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/math.h"
#include "sdfg/data_flow/library_nodes/math/ml/conv.h"
#include "sdfg/visualizer/dot_visualizer.h"

using namespace sdfg;

TEST(MathTest, ReLU) {
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
    auto& relu_node = static_cast<
        math::ml::ReLUNode&>(builder.add_library_node<math::ml::ReLUNode>(block, DebugInfo(), "output", "input"));

    builder.add_memlet(
        block,
        input_node,
        "void",
        relu_node,
        "input",
        {symbolic::integer(0), symbolic::integer(0)},
        {symbolic::integer(10), symbolic::integer(20)},
        block.debug_info()
    );
    builder.add_memlet(
        block,
        relu_node,
        "output",
        output_node,
        "void",
        {symbolic::integer(0), symbolic::integer(0)},
        {symbolic::integer(10), symbolic::integer(20)},
        block.debug_info()
    );

    EXPECT_EQ(block.dataflow().nodes().size(), 3);

    analysis::AnalysisManager analysis_manager(sdfg);
    EXPECT_TRUE(relu_node.expand(builder, analysis_manager));

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
    EXPECT_EQ(tasklet->inputs().at(0).first, "0");
    EXPECT_EQ(tasklet->inputs().at(1).first, "_in");
    EXPECT_EQ(tasklet->output().first, "_out");
    EXPECT_EQ(tasklet->output().second.primitive_type(), types::PrimitiveType::Double);
}

TEST(MathTest, Conv_2D) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();

    // Define scalar and tensor descriptors
    types::Scalar element(types::PrimitiveType::Double);

    // Input tensor X : [N=1][C=1][H=4][W=4]
    types::Array w_dim(element, symbolic::integer(4)); // W
    types::Array h_dim(w_dim, symbolic::integer(4)); // H
    types::Array c_dim(h_dim, symbolic::integer(1)); // C
    types::Array x_desc(c_dim, symbolic::integer(1)); // N

    // Weight tensor W : [M=1][C=1][KH=3][KW=3]
    types::Array kw_dim(element, symbolic::integer(3)); // KW
    types::Array kh_dim(kw_dim, symbolic::integer(3)); // KH
    types::Array c_w_dim(kh_dim, symbolic::integer(1)); // C
    types::Array w_desc(c_w_dim, symbolic::integer(1)); // M

    // Output tensor Y : [N=1][M=1][OH=2][OW=2]
    types::Array ow_dim(element, symbolic::integer(2)); // OW
    types::Array oh_dim(ow_dim, symbolic::integer(2)); // OH
    types::Array m_dim(oh_dim, symbolic::integer(1)); // M
    types::Array y_desc(m_dim, symbolic::integer(1)); // N

    // Add containers
    builder.add_container("X", x_desc);
    builder.add_container("W", w_desc);
    builder.add_container("Y", y_desc);

    // Add block and access nodes
    auto& block = builder.add_block(sdfg.root());
    auto& X_acc = builder.add_access(block, "X");
    auto& W_acc = builder.add_access(block, "W");
    auto& Y_acc = builder.add_access(block, "Y");

    // Conv parameters
    bool has_bias = false;
    std::vector<size_t> dilations = {1, 1};
    std::vector<size_t> kernel_shape = {3, 3};
    std::vector<size_t> pads = {0, 0, 0, 0};
    std::vector<size_t> strides = {1, 1};

    auto& conv_node = static_cast<
        math::ml::ConvNode&>(builder.add_library_node<
                             math::ml::ConvNode>(block, DebugInfo(), has_bias, dilations, kernel_shape, pads, strides));

    // Memlet subsets
    data_flow::Subset x_begin{symbolic::integer(0), symbolic::integer(0), symbolic::integer(0), symbolic::integer(0)};
    data_flow::Subset x_end{symbolic::integer(0), symbolic::integer(0), symbolic::integer(3), symbolic::integer(3)};

    data_flow::Subset w_begin{symbolic::integer(0), symbolic::integer(0), symbolic::integer(0), symbolic::integer(0)};
    data_flow::Subset w_end{symbolic::integer(0), symbolic::integer(0), symbolic::integer(2), symbolic::integer(2)};

    data_flow::Subset y_begin{symbolic::integer(0), symbolic::integer(0), symbolic::integer(0), symbolic::integer(0)};
    data_flow::Subset y_end{symbolic::integer(0), symbolic::integer(0), symbolic::integer(1), symbolic::integer(1)};

    // Connect memlets
    builder.add_computational_memlet(block, X_acc, conv_node, "X", x_begin, x_end, block.debug_info());
    builder.add_computational_memlet(block, W_acc, conv_node, "W", w_begin, w_end, block.debug_info());
    builder.add_computational_memlet(block, conv_node, "Y", Y_acc, y_begin, y_end, block.debug_info());

    EXPECT_EQ(block.dataflow().nodes().size(), 4);

    analysis::AnalysisManager analysis_manager(sdfg);
    EXPECT_TRUE(conv_node.expand(builder, analysis_manager));
}

TEST(MathTest, Conv_2D_Strides) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();

    // Define scalar and tensor descriptors
    types::Scalar element(types::PrimitiveType::Double);

    // Input tensor X : [N=1][C=1][H=4][W=4]
    types::Array w_dim(element, symbolic::integer(5)); // W
    types::Array h_dim(w_dim, symbolic::integer(5)); // H
    types::Array c_dim(h_dim, symbolic::integer(1)); // C
    types::Array x_desc(c_dim, symbolic::integer(1)); // N

    // Weight tensor W : [M=1][C=1][KH=3][KW=3]
    types::Array kw_dim(element, symbolic::integer(3)); // KW
    types::Array kh_dim(kw_dim, symbolic::integer(3)); // KH
    types::Array c_w_dim(kh_dim, symbolic::integer(1)); // C
    types::Array w_desc(c_w_dim, symbolic::integer(1)); // M

    // Output tensor Y : [N=1][M=1][OH=2][OW=2]
    types::Array ow_dim(element, symbolic::integer(2)); // OW
    types::Array oh_dim(ow_dim, symbolic::integer(2)); // OH
    types::Array m_dim(oh_dim, symbolic::integer(1)); // M
    types::Array y_desc(m_dim, symbolic::integer(1)); // N

    // Add containers
    builder.add_container("X", x_desc);
    builder.add_container("W", w_desc);
    builder.add_container("Y", y_desc);

    // Add block and access nodes
    auto& block = builder.add_block(sdfg.root());
    auto& X_acc = builder.add_access(block, "X");
    auto& W_acc = builder.add_access(block, "W");
    auto& Y_acc = builder.add_access(block, "Y");

    // Conv parameters
    bool has_bias = false;
    std::vector<size_t> dilations = {1, 1};
    std::vector<size_t> kernel_shape = {3, 3};
    std::vector<size_t> pads = {0, 0, 0, 0};
    std::vector<size_t> strides = {2, 2};

    auto& conv_node = static_cast<
        math::ml::ConvNode&>(builder.add_library_node<
                             math::ml::ConvNode>(block, DebugInfo(), has_bias, dilations, kernel_shape, pads, strides));

    // Memlet subsets
    data_flow::Subset x_begin{symbolic::integer(0), symbolic::integer(0), symbolic::integer(0), symbolic::integer(0)};
    data_flow::Subset x_end{symbolic::integer(0), symbolic::integer(0), symbolic::integer(4), symbolic::integer(4)};

    data_flow::Subset w_begin{symbolic::integer(0), symbolic::integer(0), symbolic::integer(0), symbolic::integer(0)};
    data_flow::Subset w_end{symbolic::integer(0), symbolic::integer(0), symbolic::integer(2), symbolic::integer(2)};

    data_flow::Subset y_begin{symbolic::integer(0), symbolic::integer(0), symbolic::integer(0), symbolic::integer(0)};
    data_flow::Subset y_end{symbolic::integer(0), symbolic::integer(0), symbolic::integer(1), symbolic::integer(1)};

    // Connect memlets
    builder.add_computational_memlet(block, X_acc, conv_node, "X", x_begin, x_end, block.debug_info());
    builder.add_computational_memlet(block, W_acc, conv_node, "W", w_begin, w_end, block.debug_info());
    builder.add_computational_memlet(block, conv_node, "Y", Y_acc, y_begin, y_end, block.debug_info());

    EXPECT_EQ(block.dataflow().nodes().size(), 4);

    analysis::AnalysisManager analysis_manager(sdfg);
    EXPECT_TRUE(conv_node.expand(builder, analysis_manager));
}
