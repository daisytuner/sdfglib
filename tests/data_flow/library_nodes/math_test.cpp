#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/blas/blas.h"
#include "sdfg/data_flow/library_nodes/math/math.h"
#include "sdfg/data_flow/library_nodes/math/ml/conv.h"
#include "sdfg/data_flow/library_nodes/math/ml/maxpool.h"

#include "sdfg/data_flow/library_nodes/math/blas/gemm.h"
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

    builder.add_computational_memlet(
        block,
        input_node,
        relu_node,
        "input",
        {symbolic::integer(0), symbolic::integer(0)},
        {symbolic::integer(10), symbolic::integer(20)},
        array_desc_2,
        block.debug_info()
    );
    builder.add_computational_memlet(
        block,
        relu_node,
        "output",
        output_node,
        {symbolic::integer(0), symbolic::integer(0)},
        {symbolic::integer(10), symbolic::integer(20)},
        array_desc_2,
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
    EXPECT_EQ(tasklet->inputs().at(0), "0");
    EXPECT_EQ(tasklet->inputs().at(1), "_in");
    EXPECT_EQ(tasklet->output(), "_out");
}

TEST(MathTest, Gemm) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();

    int dim_i = 10;
    int dim_j = 20;
    int dim_k = 30;

    // res: ixj, A: ixk, B: kxj

    types::Scalar desc(types::PrimitiveType::Float);
    types::Array arr_a_type(desc, symbolic::mul(symbolic::integer(dim_k), symbolic::integer(dim_i)));
    types::Array arr_b_type(desc, symbolic::mul(symbolic::integer(dim_j), symbolic::integer(dim_k)));
    types::Array arr_res_type(desc, symbolic::mul(symbolic::integer(dim_j), symbolic::integer(dim_i)));

    builder.add_container("arr_a", arr_a_type);
    builder.add_container("arr_b", arr_b_type);
    builder.add_container("output", arr_res_type);

    auto& block = builder.add_block(sdfg.root());

    auto& input_a_node = builder.add_access(block, "arr_a");
    auto& input_b_node = builder.add_access(block, "arr_b");
    auto c_var_name = "output";
    auto& dummy_input_node = builder.add_access(block, c_var_name);
    auto& output_node = builder.add_access(block, c_var_name);
    auto& gemm_node = static_cast<math::blas::GEMMNode&>(builder.add_library_node<math::blas::GEMMNode>(
        block,
        DebugInfo(),
        data_flow::ImplementationType_NONE,
        math::blas::BLAS_Precision::s,
        math::blas::BLAS_Layout::RowMajor,
        math::blas::BLAS_Transpose::No,
        math::blas::BLAS_Transpose::No,
        symbolic::integer(dim_i),
        symbolic::integer(dim_j),
        symbolic::integer(dim_k),
        symbolic::integer(dim_j), // lda
        symbolic::integer(dim_k), // ldb
        symbolic::integer(dim_j), // ldc
        "1", // alpha
        "0" // beta
    ));

    builder.add_computational_memlet(
        block,
        input_a_node,
        gemm_node,
        "A",
        {symbolic::integer(0)},
        {symbolic::mul(symbolic::integer(dim_i), symbolic::integer(dim_k))},
        arr_a_type
    );
    builder.add_computational_memlet(
        block,
        input_b_node,
        gemm_node,
        "B",
        {symbolic::integer(0)},
        {symbolic::mul(symbolic::integer(dim_k), symbolic::integer(dim_j))},
        arr_b_type
    );
    builder.add_computational_memlet(
        block,
        dummy_input_node,
        gemm_node,
        "C",
        {symbolic::integer(0)},
        {symbolic::mul(symbolic::integer(dim_i), symbolic::integer(dim_j))},
        arr_res_type
    );

    builder.add_computational_memlet(
        block,
        gemm_node,
        "C",
        output_node,
        {symbolic::integer(0)},
        {symbolic::mul(symbolic::integer(dim_i), symbolic::integer(dim_j))},
        arr_res_type
    );

    EXPECT_EQ(block.dataflow().nodes().size(), 5);

    std::filesystem::path before_file = "gemm.before-expand.sdfg.dot";
    visualizer::DotVisualizer::writeToFile(builder.subject(), &before_file);

    analysis::AnalysisManager analysis_manager(sdfg);
    EXPECT_TRUE(gemm_node.expand(builder, analysis_manager));
    builder.subject().validate();

    std::filesystem::path after_file = "gemm.after-expand.sdfg.dot";
    visualizer::DotVisualizer::writeToFile(builder.subject(), &after_file);

    EXPECT_EQ(sdfg.root().size(), 1);
    auto new_sequence = dynamic_cast<structured_control_flow::Sequence*>(&sdfg.root().at(0).first);
    EXPECT_NE(new_sequence, nullptr);

    auto map_1 = dynamic_cast<structured_control_flow::Map*>(&new_sequence->at(0).first);
    EXPECT_NE(map_1, nullptr);
    EXPECT_EQ(map_1->root().size(), 1);

    auto map_2 = dynamic_cast<structured_control_flow::Map*>(&map_1->root().at(0).first);
    EXPECT_NE(map_2, nullptr);
    EXPECT_EQ(map_2->root().size(), 3);

    auto block_init = dynamic_cast<structured_control_flow::Block*>(&map_2->root().at(0).first);
    EXPECT_NE(block_init, nullptr);
    EXPECT_EQ(block_init->dataflow().nodes().size(), 2);
    auto init_tasklet = *block_init->dataflow().tasklets().begin();
    EXPECT_EQ(init_tasklet->code(), data_flow::TaskletCode::assign);
    EXPECT_EQ(init_tasklet->inputs().at(0), "0.0");
    EXPECT_EQ(init_tasklet->output(), "_out");

    auto map_3 = dynamic_cast<structured_control_flow::Map*>(&map_2->root().at(1).first);
    EXPECT_NE(map_3, nullptr);
    EXPECT_EQ(map_3->root().size(), 1);

    auto block_fma = dynamic_cast<structured_control_flow::Block*>(&map_3->root().at(0).first);
    EXPECT_NE(block_fma, nullptr);
    EXPECT_EQ(block_fma->dataflow().nodes().size(), 5);

    auto tasklet = *block_fma->dataflow().tasklets().begin();
    EXPECT_EQ(tasklet->code(), data_flow::TaskletCode::fma);
    EXPECT_EQ(tasklet->inputs().size(), 3);
    EXPECT_EQ(tasklet->inputs().at(0), "_in1");
    EXPECT_EQ(tasklet->inputs().at(1), "_in2");
    EXPECT_EQ(tasklet->inputs().at(2), "_in3");
    EXPECT_EQ(tasklet->output(), "_out");

    auto block_flush = dynamic_cast<structured_control_flow::Block*>(&map_2->root().at(2).first);
    EXPECT_NE(block_flush, nullptr);
    EXPECT_EQ(block_flush->dataflow().nodes().size(), 8);
    auto flush_tasklets = block_flush->dataflow().tasklets();
    EXPECT_EQ(flush_tasklets.size(), 3);
    for (auto* tasklet : flush_tasklets) {
        if (tasklet->code() == data_flow::add) {
            EXPECT_EQ(tasklet->output(), "_out");
            auto& final_edge = *block_flush->dataflow().out_edges(*tasklet).begin();
            auto* final_access = dynamic_cast<data_flow::AccessNode*>(&final_edge.dst());
            EXPECT_NE(final_access, nullptr);
            EXPECT_EQ(final_access->data(), c_var_name);
        }
    }
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
    builder.add_computational_memlet(block, X_acc, conv_node, "X", x_begin, x_end, x_desc, block.debug_info());
    builder.add_computational_memlet(block, W_acc, conv_node, "W", w_begin, w_end, w_desc, block.debug_info());
    builder.add_computational_memlet(block, conv_node, "Y", Y_acc, y_begin, y_end, y_desc, block.debug_info());

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
    builder.add_computational_memlet(block, X_acc, conv_node, "X", x_begin, x_end, x_desc, block.debug_info());
    builder.add_computational_memlet(block, W_acc, conv_node, "W", w_begin, w_end, w_desc, block.debug_info());
    builder.add_computational_memlet(block, conv_node, "Y", Y_acc, y_begin, y_end, y_desc, block.debug_info());

    EXPECT_EQ(block.dataflow().nodes().size(), 4);

    analysis::AnalysisManager analysis_manager(sdfg);
    EXPECT_TRUE(conv_node.expand(builder, analysis_manager));
}

TEST(MathTest, MaxPool_2D) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();

    // Define scalar and tensor descriptors
    types::Scalar element(types::PrimitiveType::Double);

    // Input tensor X: [N=1][C=1][H=4][W=4]
    types::Array w_dim(element, symbolic::integer(4)); // W
    types::Array h_dim(w_dim, symbolic::integer(4)); // H
    types::Array c_dim(h_dim, symbolic::integer(1)); // C
    types::Array x_desc(c_dim, symbolic::integer(1)); // N

    // Output tensor Y after 2x2 maxpool with stride 2 -> [N=1][C=1][OH=2][OW=2]
    types::Array ow_dim(element, symbolic::integer(2)); // OW
    types::Array oh_dim(ow_dim, symbolic::integer(2)); // OH
    types::Array c_out_dim(oh_dim, symbolic::integer(1)); // C
    types::Array y_desc(c_out_dim, symbolic::integer(1)); // N

    // Add containers
    builder.add_container("X", x_desc);
    builder.add_container("Y", y_desc);

    // Add block and access nodes
    auto& block = builder.add_block(sdfg.root());
    auto& X_acc = builder.add_access(block, "X");
    auto& Y_acc = builder.add_access(block, "Y");

    // MaxPool parameters
    std::vector<size_t> kernel_shape = {2, 2};
    std::vector<size_t> pads = {0, 0, 0, 0};
    std::vector<size_t> strides = {2, 2};

    auto& pool_node =
        static_cast<math::ml::MaxPoolNode&>(builder.add_library_node<
                                            math::ml::MaxPoolNode>(block, DebugInfo(), kernel_shape, pads, strides));

    // Subsets
    data_flow::Subset x_begin{symbolic::integer(0), symbolic::integer(0), symbolic::integer(0), symbolic::integer(0)};
    data_flow::Subset x_end{symbolic::integer(0), symbolic::integer(0), symbolic::integer(3), symbolic::integer(3)};

    data_flow::Subset y_begin{symbolic::integer(0), symbolic::integer(0), symbolic::integer(0), symbolic::integer(0)};
    data_flow::Subset y_end{symbolic::integer(0), symbolic::integer(0), symbolic::integer(1), symbolic::integer(1)};

    // Memlets
    builder.add_computational_memlet(block, X_acc, pool_node, "X", x_begin, x_end, x_desc, block.debug_info());
    builder.add_computational_memlet(block, pool_node, "Y", Y_acc, y_begin, y_end, y_desc, block.debug_info());

    EXPECT_EQ(block.dataflow().nodes().size(), 3);

    analysis::AnalysisManager analysis_manager(sdfg);
    EXPECT_TRUE(pool_node.expand(builder, analysis_manager));
}

TEST(MathTest, Dot) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();

    auto n = symbolic::integer(10);
    auto stride_a = symbolic::integer(2);
    auto stride_b = symbolic::integer(2);

    types::Scalar desc(types::PrimitiveType::Double);
    types::Array array_desc(desc, n);

    builder.add_container("a", array_desc);
    builder.add_container("b", array_desc);
    builder.add_container("c", desc);

    auto& block = builder.add_block(sdfg.root());

    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");
    auto& c_node = builder.add_access(block, "c");

    auto& dot_node = static_cast<math::blas::DotNode&>(builder.add_library_node<math::blas::DotNode>(
        block, DebugInfo(), math::blas::ImplementationType_BLAS, math::blas::BLAS_Precision::d, n, stride_a, stride_b
    ));

    builder.add_computational_memlet(
        block,
        a_node,
        dot_node,
        "x",
        {symbolic::zero()},
        {symbolic::sub(n, symbolic::integer(1))},
        array_desc,
        block.debug_info()
    );
    builder.add_computational_memlet(
        block,
        b_node,
        dot_node,
        "y",
        {symbolic::zero()},
        {symbolic::sub(n, symbolic::integer(1))},
        array_desc,
        block.debug_info()
    );
    builder.add_computational_memlet(block, dot_node, "res", c_node, {}, {}, desc, block.debug_info());

    EXPECT_EQ(block.dataflow().nodes().size(), 4);

    analysis::AnalysisManager analysis_manager(sdfg);
    EXPECT_TRUE(dot_node.expand(builder, analysis_manager));
}
