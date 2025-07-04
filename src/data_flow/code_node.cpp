#include "sdfg/data_flow/code_node.h"

namespace sdfg {
namespace data_flow {

CodeNode::CodeNode(size_t element_id, const DebugInfo& debug_info, const graph::Vertex vertex, DataFlowGraph& parent)
    : DataFlowNode(element_id, debug_info, vertex, parent) {};

} // namespace data_flow
} // namespace sdfg
