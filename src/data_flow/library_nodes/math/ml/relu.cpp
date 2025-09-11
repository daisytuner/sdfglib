#include "sdfg/data_flow/library_nodes/math/ml/relu.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

#include "sdfg/analysis/scope_analysis.h"

namespace sdfg {
namespace math {
namespace ml {

ReLUNode::ReLUNode(
    size_t element_id,
    const DebugInfoRegion& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<symbolic::Expression>& shape
)
    : ElementWiseUnaryNode(element_id, debug_info, vertex, parent, LibraryNodeType_ReLU, shape, {}) {}

bool ReLUNode::expand_operation(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Sequence& body,
    const std::string& input_name,
    const std::string& output_name,
    const types::IType& input_type,
    const types::IType& output_type,
    const data_flow::Subset& subset
) {
    // Add code
    types::Scalar base_type(input_type.primitive_type());

    auto& code_block = builder.add_block(body);
    auto& input_node_new = builder.add_access(code_block, input_name);
    auto& output_node_new = builder.add_access(code_block, output_name);
    auto& zero_node = builder.add_constant(code_block, "0.0", base_type);
    auto& tasklet = builder.add_tasklet(code_block, data_flow::TaskletCode::max, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(code_block, zero_node, tasklet, "_in1", {}, base_type);
    builder.add_computational_memlet(code_block, input_node_new, tasklet, "_in2", subset, input_type);
    builder.add_computational_memlet(code_block, tasklet, "_out", output_node_new, subset, output_type);

    return true;
}

std::unique_ptr<data_flow::DataFlowNode> ReLUNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<
        data_flow::DataFlowNode>(new ReLUNode(element_id, this->debug_info(), vertex, parent, this->shape_));
}

} // namespace ml
} // namespace math
} // namespace sdfg
