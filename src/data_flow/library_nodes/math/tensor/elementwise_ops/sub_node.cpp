#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/sub_node.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

#include "sdfg/analysis/scope_analysis.h"

namespace sdfg {
namespace math {
namespace tensor {

SubNode::SubNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<symbolic::Expression>& shape
)
    : ElementWiseBinaryNode(element_id, debug_info, vertex, parent, LibraryNodeType_Sub, shape) {}

bool SubNode::expand_operation(
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

    auto& tasklet = builder.add_tasklet(code_block, data_flow::TaskletCode::fp_sub, "_out", {"_in1", "_in2"});

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

    return true;
}

std::unique_ptr<data_flow::DataFlowNode> SubNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<
        data_flow::DataFlowNode>(new SubNode(element_id, this->debug_info(), vertex, parent, this->shape_));
}

} // namespace tensor
} // namespace math
} // namespace sdfg
