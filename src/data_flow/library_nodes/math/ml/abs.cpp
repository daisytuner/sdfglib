#include "sdfg/data_flow/library_nodes/math/ml/abs.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

#include "sdfg/analysis/scope_analysis.h"

#include "sdfg/data_flow/library_nodes/math/intrinsic.h"

namespace sdfg {
namespace math {
namespace ml {

AbsNode::AbsNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<symbolic::Expression>& shape
)
    : ElementWiseUnaryNode(element_id, debug_info, vertex, parent, LibraryNodeType_Abs, shape) {}

bool AbsNode::expand_operation(
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
    
    auto& libnode = builder.add_library_node<math::IntrinsicNode>(code_block, body.debug_info(), "fabs", 1);
    
    builder.add_computational_memlet(code_block, input_node_new, libnode, "_in", subset, input_type);
    builder.add_computational_memlet(code_block, libnode, "_out", output_node_new, subset, output_type);

    return true;
}

std::unique_ptr<data_flow::DataFlowNode> AbsNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<
        data_flow::DataFlowNode>(new AbsNode(element_id, this->debug_info(), vertex, parent, this->shape_));
}

} // namespace ml
} // namespace math
} // namespace sdfg
