#include "sdfg/data_flow/data_flow_node.h"

using json = nlohmann::json;

namespace sdfg {
namespace data_flow {

DataFlowNode::DataFlowNode(const DebugInfo& debug_info, const graph::Vertex vertex,
                           DataFlowGraph& parent)
    : Element(debug_info), vertex_(vertex), parent_(&parent) {

      };

graph::Vertex DataFlowNode::vertex() const { return this->vertex_; };

const DataFlowGraph& DataFlowNode::get_parent() const { return *this->parent_; };

DataFlowGraph& DataFlowNode::get_parent() { return *this->parent_; };

}  // namespace data_flow
}  // namespace sdfg
