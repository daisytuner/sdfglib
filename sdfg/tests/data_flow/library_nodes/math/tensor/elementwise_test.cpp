#include "gtest/gtest.h"
#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

#include "sdfg/data_flow/library_nodes/math/tensor/elementwise/cast_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise/elementwise_node.h"
#include "sdfg/passes/schedules/expansion_pass.h"

using namespace sdfg;

template<typename NodeType>
void TestUnary(std::vector<size_t> shape_dims) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Double);
    types::Pointer desc_ptr(desc);

    builder.add_container("a", desc_ptr);
    builder.add_container("b", desc_ptr);

    auto& block = builder.add_block(sdfg.root());

    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");

    std::vector<symbolic::Expression> shape;
    for (auto d : shape_dims) {
        shape.push_back(symbolic::integer(d));
    }
    types::Tensor tensor_desc(desc, shape);

    auto& node =
        static_cast<NodeType&>(builder.add_library_node<NodeType>(block, DebugInfo(), data_flow::TaskletCode::fp_neg));

    builder.add_computational_memlet(block, a_node, node, "_in1", {}, tensor_desc, block.debug_info());
    builder.add_computational_memlet(block, node, "_out", b_node, {}, tensor_desc, block.debug_info());

    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::ExpansionPass expansion_pass;
    bool applies = false;
    do {
        applies = expansion_pass.run(builder, analysis_manager);
    } while (applies);

    auto& new_sequence = dynamic_cast<structured_control_flow::Sequence&>(sdfg.root().at(0).first);

    // Navigate to the innermost map
    structured_control_flow::Sequence* current_scope = &new_sequence;
    for (size_t i = 0; i < shape_dims.size(); ++i) {
        auto map_loop = dynamic_cast<structured_control_flow::Map*>(&current_scope->at(0).first);
        ASSERT_NE(map_loop, nullptr);
        current_scope = &map_loop->root();
    }

    auto code_block = dynamic_cast<structured_control_flow::Block*>(&current_scope->at(0).first);
    ASSERT_NE(code_block, nullptr);

    // Check that the block is not empty (contains either tasklets or library nodes)
    bool has_content = !code_block->dataflow().tasklets().empty() || !code_block->dataflow().library_nodes().empty();
    EXPECT_TRUE(has_content) << "Inner block is empty for " << typeid(NodeType).name();

    // Check subsets of the first node's edges
    data_flow::DataFlowNode* inner_node = nullptr;
    if (!code_block->dataflow().library_nodes().empty()) {
        inner_node = *code_block->dataflow().library_nodes().begin();
    } else if (!code_block->dataflow().tasklets().empty()) {
        inner_node = *code_block->dataflow().tasklets().begin();
    }
    ASSERT_NE(inner_node, nullptr);

    auto& dataflow = inner_node->get_parent();

    // Check input edges
    for (auto& edge : dataflow.in_edges(*inner_node)) {
        if (dynamic_cast<const data_flow::ConstantNode*>(&edge.src()) != nullptr) {
            continue; // Skip constant nodes which may have different subset sizes
        }
        if (edge.subset().size() != shape_dims.size()) {
            EXPECT_EQ(edge.subset().size(), shape_dims.size())
                << "Input subset size is not correct for " << typeid(NodeType).name();
        }
    }

    // Check output edges
    for (auto& edge : dataflow.out_edges(*inner_node)) {
        if (edge.subset().size() != shape_dims.size()) {
            EXPECT_EQ(edge.subset().size(), shape_dims.size())
                << "Output subset size is not correct for " << typeid(NodeType).name();
        }
    }
}

#define REGISTER_ELEMENTWISE_TEST(NodeType, Dim)          \
    TEST(ElementWiseTest, NodeType##_##Dim##D) {          \
        std::vector<size_t> dims;                         \
        for (int i = 0; i < Dim; ++i) dims.push_back(32); \
        TestUnary<math::tensor::NodeType>(dims);          \
    }

REGISTER_ELEMENTWISE_TEST(ElementwiseNode, 1)
REGISTER_ELEMENTWISE_TEST(ElementwiseNode, 2)
REGISTER_ELEMENTWISE_TEST(ElementwiseNode, 3)
REGISTER_ELEMENTWISE_TEST(ElementwiseNode, 4)

// Cast tests - specialized template for CastNode
template<types::PrimitiveType SourceType, types::PrimitiveType TargetType>
void TestCast(std::vector<size_t> shape_dims) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar source_desc(SourceType);
    types::Scalar target_desc(TargetType);
    types::Pointer source_ptr(source_desc);
    types::Pointer target_ptr(target_desc);

    builder.add_container("a", source_ptr);
    builder.add_container("b", target_ptr);

    auto& block = builder.add_block(sdfg.root());

    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");

    std::vector<symbolic::Expression> shape;
    for (auto d : shape_dims) {
        shape.push_back(symbolic::integer(d));
    }
    types::Tensor src_tensor_desc(source_desc, shape);
    types::Tensor dst_tensor_desc(target_desc, shape);

    auto& node =
        static_cast<math::tensor::CastNode&>(builder.add_library_node<math::tensor::CastNode>(block, DebugInfo()));

    builder.add_computational_memlet(block, a_node, node, "_in1", {}, src_tensor_desc, block.debug_info());
    builder.add_computational_memlet(block, node, "_out", b_node, {}, dst_tensor_desc, block.debug_info());

    analysis::AnalysisManager analysis_manager(sdfg);
    EXPECT_TRUE(node.expand(builder, analysis_manager));

    auto& new_sequence = dynamic_cast<structured_control_flow::Sequence&>(sdfg.root().at(0).first);

    // Navigate to the innermost map
    structured_control_flow::Sequence* current_scope = &new_sequence;
    for (size_t i = 0; i < shape_dims.size(); ++i) {
        auto map_loop = dynamic_cast<structured_control_flow::Map*>(&current_scope->at(0).first);
        ASSERT_NE(map_loop, nullptr);
        current_scope = &map_loop->root();
    }

    auto code_block = dynamic_cast<structured_control_flow::Block*>(&current_scope->at(0).first);
    ASSERT_NE(code_block, nullptr);

    // Check that the block is not empty (contains either tasklets or library nodes)
    bool has_content = !code_block->dataflow().tasklets().empty() || !code_block->dataflow().library_nodes().empty();
    EXPECT_TRUE(has_content) << "Inner block is empty for CastNode";

    // Check subsets of the first node's edges
    data_flow::DataFlowNode* inner_node = nullptr;
    if (!code_block->dataflow().library_nodes().empty()) {
        inner_node = *code_block->dataflow().library_nodes().begin();
    } else if (!code_block->dataflow().tasklets().empty()) {
        inner_node = *code_block->dataflow().tasklets().begin();
    }
    ASSERT_NE(inner_node, nullptr);

    auto& dataflow = inner_node->get_parent();

    // Check input edges
    for (auto& edge : dataflow.in_edges(*inner_node)) {
        if (edge.subset().size() != shape_dims.size()) {
            EXPECT_EQ(edge.subset().size(), 1) << "Input subset size is not 1 for CastNode";
        }
        // Check that input type is the source type
        EXPECT_EQ(edge.result_type(sdfg)->primitive_type(), SourceType);
    }

    // Check output edges
    for (auto& edge : dataflow.out_edges(*inner_node)) {
        if (edge.subset().size() != shape_dims.size()) {
            EXPECT_EQ(edge.subset().size(), 1) << "Output subset size is not 1 for CastNode";
        }
        // Check that output type is the target type
        EXPECT_EQ(edge.result_type(sdfg)->primitive_type(), TargetType);
    }
}

#define REGISTER_CAST_TEST(SourceType, TargetType, Dim)                                     \
    TEST(ElementWiseTest, CastNode_##SourceType##_to_##TargetType##_##Dim##D) {             \
        std::vector<size_t> dims;                                                           \
        for (int i = 0; i < Dim; ++i) dims.push_back(32);                                   \
        TestCast<types::PrimitiveType::SourceType, types::PrimitiveType::TargetType>(dims); \
    }

// Register cast tests for various type conversions
REGISTER_CAST_TEST(Int32, Float, 1)
REGISTER_CAST_TEST(Int32, Float, 2)
REGISTER_CAST_TEST(Int32, Float, 3)
REGISTER_CAST_TEST(Int32, Float, 4)

REGISTER_CAST_TEST(Float, Int32, 1)
REGISTER_CAST_TEST(Float, Int32, 2)
REGISTER_CAST_TEST(Float, Int32, 3)
REGISTER_CAST_TEST(Float, Int32, 4)

REGISTER_CAST_TEST(Float, Double, 1)
REGISTER_CAST_TEST(Float, Double, 2)
REGISTER_CAST_TEST(Float, Double, 3)
REGISTER_CAST_TEST(Float, Double, 4)

REGISTER_CAST_TEST(Double, Float, 1)
REGISTER_CAST_TEST(Double, Float, 2)
REGISTER_CAST_TEST(Double, Float, 3)
REGISTER_CAST_TEST(Double, Float, 4)

REGISTER_CAST_TEST(Int32, Int64, 1)
REGISTER_CAST_TEST(Int32, Int64, 2)
REGISTER_CAST_TEST(Int32, Int64, 3)
REGISTER_CAST_TEST(Int32, Int64, 4)

REGISTER_CAST_TEST(Int64, Int32, 1)
REGISTER_CAST_TEST(Int64, Int32, 2)
REGISTER_CAST_TEST(Int64, Int32, 3)
REGISTER_CAST_TEST(Int64, Int32, 4)
