#include "sdfg/data_flow/library_nodes/math/ml/relu.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

#include "sdfg/analysis/scope_analysis.h"

namespace sdfg {
namespace math {
namespace ml {

ReLUNode::ReLUNode(
    size_t element_id, const DebugInfoRegion& debug_info, const graph::Vertex vertex, data_flow::DataFlowGraph& parent
)
    : ElementWiseUnaryNode(element_id, debug_info, vertex, parent, LibraryNodeType_ReLU, {}) {}

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
    auto& code_block = builder.add_block(body);
    auto& input_node_new = builder.add_access(code_block, input_name);
    auto& output_node_new = builder.add_access(code_block, output_name);
    auto& tasklet = builder.add_tasklet(code_block, data_flow::TaskletCode::max, "_out", {"0.0f", "_in"});
    builder.add_computational_memlet(code_block, input_node_new, tasklet, "_in", subset, input_type);
    builder.add_computational_memlet(code_block, tasklet, "_out", output_node_new, subset, output_type);

    return true;
}

std::unique_ptr<data_flow::DataFlowNode> ReLUNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(new ReLUNode(element_id, this->debug_info(), vertex, parent));
}

} // namespace ml
} // namespace math
} // namespace sdfg
