#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/minimum_node.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_node.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace math {
namespace tensor {

MinimumNode::MinimumNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<symbolic::Expression>& shape
)
    : ElementWiseBinaryNode(element_id, debug_info, vertex, parent, LibraryNodeType_Minimum, shape) {}

bool MinimumNode::expand_operation(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Sequence& body,
    const std::string& input_name_a,
    const std::string& input_name_b,
    const std::string& output_name,
    const types::IType& input_type_a,
    const types::IType& input_type_b,
    const types::IType& output_type,
    const data_flow::Subset& subset
) {
    auto& code_block = builder.add_block(body);
    data_flow::AccessNode* input_node_a;
    if (builder.subject().exists(input_name_a)) {
        input_node_a = &builder.add_access(code_block, input_name_a);
    } else {
        input_node_a = &builder.add_constant(code_block, input_name_a, input_type_a);
    }
    data_flow::AccessNode* input_node_b;
    if (builder.subject().exists(input_name_b)) {
        input_node_b = &builder.add_access(code_block, input_name_b);
    } else {
        input_node_b = &builder.add_constant(code_block, input_name_b, input_type_b);
    }
    auto& output_node = builder.add_access(code_block, output_name);

    bool is_int = types::is_integer(input_type_a.primitive_type());

    if (is_int) {
        // Use tasklets for integer types - distinguish between signed and unsigned
        bool is_signed_int = types::is_signed(input_type_a.primitive_type());
        auto tasklet_code = is_signed_int ? data_flow::TaskletCode::int_smin : data_flow::TaskletCode::int_umin;
        auto& tasklet = builder.add_tasklet(code_block, tasklet_code, "_out", {"_in1", "_in2"});

        if (input_type_a.type_id() == types::TypeID::Scalar) {
            builder.add_computational_memlet(code_block, *input_node_a, tasklet, "_in1", {}, input_type_a);
        } else {
            builder.add_computational_memlet(code_block, *input_node_a, tasklet, "_in1", subset, input_type_a);
        }
        if (input_type_b.type_id() == types::TypeID::Scalar) {
            builder.add_computational_memlet(code_block, *input_node_b, tasklet, "_in2", {}, input_type_b);
        } else {
            builder.add_computational_memlet(code_block, *input_node_b, tasklet, "_in2", subset, input_type_b);
        }
        builder.add_computational_memlet(code_block, tasklet, "_out", output_node, subset, output_type);
    } else {
        // Use intrinsics for floating-point types with correct suffix
        std::string intrinsic_name = TensorNode::get_intrinsic_name("fmin", input_type_a.primitive_type());
        auto& node = builder.add_library_node<cmath::CMathNode>(code_block, this->debug_info(), intrinsic_name, 2);

        if (input_type_a.type_id() == types::TypeID::Scalar) {
            builder.add_computational_memlet(code_block, *input_node_a, node, "_in1", {}, input_type_a, DebugInfo());
        } else {
            builder.add_computational_memlet(code_block, *input_node_a, node, "_in1", subset, input_type_a, DebugInfo());
        }
        if (input_type_b.type_id() == types::TypeID::Scalar) {
            builder.add_computational_memlet(code_block, *input_node_b, node, "_in2", {}, input_type_b, DebugInfo());
        } else {
            builder.add_computational_memlet(code_block, *input_node_b, node, "_in2", subset, input_type_b, DebugInfo());
        }
        builder.add_computational_memlet(code_block, node, "_out", output_node, subset, output_type, DebugInfo());
    }

    return true;
}

std::unique_ptr<data_flow::DataFlowNode> MinimumNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<
        data_flow::DataFlowNode>(new MinimumNode(element_id, this->debug_info(), vertex, parent, this->shape_));
}

} // namespace tensor
} // namespace math
} // namespace sdfg
