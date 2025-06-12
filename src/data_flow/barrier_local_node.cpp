#include "sdfg/data_flow/barrier_local_node.h"

#include "sdfg/data_flow/library_node.h"

namespace sdfg {
namespace data_flow {

BarrierLocalNode::BarrierLocalNode(const DebugInfo& debug_info, const graph::Vertex vertex,
                                   DataFlowGraph& parent)
    : LibraryNode(debug_info, vertex, parent, LibraryNodeCode{"barrier_local"}, {}, {}, true) {

      };

const LibraryNodeCode& BarrierLocalNode::code() const { return this->code_; };

const std::vector<std::string>& BarrierLocalNode::inputs() const { return this->inputs_; };

const std::vector<std::string>& BarrierLocalNode::outputs() const { return this->outputs_; };

const std::string& BarrierLocalNode::input(size_t index) const { return this->inputs_[index]; };

const std::string& BarrierLocalNode::output(size_t index) const { return this->outputs_[index]; };

bool BarrierLocalNode::side_effect() const { return this->side_effect_; };

bool BarrierLocalNode::needs_connector(size_t index) const {
    // Is non-constant, if starts with _in prefix
    if (this->inputs_[index].compare(0, 3, "_in") == 0) {
        return true;
    }
    return false;
};

std::unique_ptr<DataFlowNode> BarrierLocalNode::clone(const graph::Vertex vertex,
                                                      DataFlowGraph& parent) const {
    return std::unique_ptr<BarrierLocalNode>(
        new BarrierLocalNode(this->debug_info_, vertex, parent));
};

void BarrierLocalNode::replace(const symbolic::Expression& old_expression,
                               const symbolic::Expression& new_expression) {
    // Do nothing
};

}  // namespace data_flow
}  // namespace sdfg
