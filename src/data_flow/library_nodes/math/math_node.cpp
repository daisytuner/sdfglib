#include "sdfg/data_flow/library_nodes/math/math_node.h"

namespace sdfg {
namespace math {

MathNode::MathNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const data_flow::LibraryNodeCode& code,
    const std::vector<std::string>& outputs,
    const std::vector<std::string>& inputs
)
    : LibraryNode(element_id, debug_info, vertex, parent, code, outputs, inputs, false) {}

} // namespace math
} // namespace sdfg
