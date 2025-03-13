#include "sdfg/data_flow/library_node.h"

#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace data_flow {

LibraryNode::LibraryNode(size_t element_id, const DebugInfo& debug_info,
                         const graph::Vertex& vertex, DataFlowGraph& parent,
                         const std::vector<std::pair<std::string, sdfg::types::Scalar>>& outputs,
                         const std::vector<std::pair<std::string, sdfg::types::Scalar>>& inputs,
                         const LibraryNodeType& call, const bool side_effect)
    : CodeNode(element_id, debug_info, vertex, parent, outputs, inputs),
      call_(call),
      side_effect_(side_effect){

      };

const LibraryNodeType& LibraryNode::call() const { return this->call_; };

const bool LibraryNode::has_side_effect() const { return this->side_effect_; };

std::unique_ptr<DataFlowNode> LibraryNode::clone(const graph::Vertex& vertex,
                                                 DataFlowGraph& parent) const {
    return std::unique_ptr<LibraryNode>(
        new LibraryNode(this->element_id_, this->debug_info_, vertex, parent, this->outputs_,
                        this->inputs_, this->call_, this->side_effect_));
};

void LibraryNode::replace(const symbolic::Expression& old_expression,
                          const symbolic::Expression& new_expression){};

}  // namespace data_flow
}  // namespace sdfg
