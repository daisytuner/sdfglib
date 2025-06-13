#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/library_nodes/device_alloc_node.h"

namespace sdfg {
namespace data_flow {

DeviceAllocNode::DeviceAllocNode(const DebugInfo& debug_info, const graph::Vertex vertex,
                                 DataFlowGraph& parent, const data_flow::LibraryNodeCode code,
                                 const std::vector<std::string>& outputs,
                                 const std::vector<std::string>& inputs, const bool side_effect)
    : LibraryNode(debug_info, vertex, parent, code, outputs, inputs, side_effect) {

      };

const LibraryNodeCode& DeviceAllocNode::code() const { return this->code_; };

const std::vector<std::string>& DeviceAllocNode::inputs() const { return this->inputs_; };

const std::vector<std::string>& DeviceAllocNode::outputs() const { return this->outputs_; };

const std::string& DeviceAllocNode::input(size_t index) const { return this->inputs_[index]; };

const std::string& DeviceAllocNode::output(size_t index) const { return this->outputs_[index]; };

bool DeviceAllocNode::side_effect() const { return this->side_effect_; };

bool DeviceAllocNode::needs_connector(size_t index) const {
    // Is non-constant, if starts with _in prefix
    if (this->inputs_[index].compare(0, 3, "_in") == 0) {
        return true;
    }
    return false;
};

std::unique_ptr<DataFlowNode> DeviceAllocNode::clone(const graph::Vertex vertex,
                                                     DataFlowGraph& parent) const {
    return std::unique_ptr<DeviceAllocNode>(new DeviceAllocNode(this->debug_info_, vertex, parent,
                                                                this->code_, this->outputs_,
                                                                this->inputs_, this->side_effect_));
};

void DeviceAllocNode::replace(const symbolic::Expression& old_expression,
                              const symbolic::Expression& new_expression) {
    // Do nothing
};

}  // namespace data_flow
}  // namespace sdfg
