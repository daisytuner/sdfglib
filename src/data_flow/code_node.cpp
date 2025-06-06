#include "sdfg/data_flow/code_node.h"

namespace sdfg {
namespace data_flow {

CodeNode::CodeNode(const DebugInfo& debug_info, const graph::Vertex vertex, DataFlowGraph& parent)
    : DataFlowNode(debug_info, vertex, parent) {};

}  // namespace data_flow
}  // namespace sdfg
