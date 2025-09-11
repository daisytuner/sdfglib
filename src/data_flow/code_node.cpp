#include "sdfg/data_flow/code_node.h"

namespace sdfg {
namespace data_flow {

CodeNode::CodeNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    DataFlowGraph& parent,
    const std::vector<std::string>& outputs,
    const std::vector<std::string>& inputs
)
    : DataFlowNode(element_id, debug_info, vertex, parent), outputs_(outputs), inputs_(inputs) {};

const std::vector<std::string>& CodeNode::outputs() const { return this->outputs_; };

const std::vector<std::string>& CodeNode::inputs() const { return this->inputs_; };

std::vector<std::string>& CodeNode::outputs() { return this->outputs_; };

std::vector<std::string>& CodeNode::inputs() { return this->inputs_; };

const std::string& CodeNode::output(size_t index) const { return this->outputs_[index]; };

const std::string& CodeNode::input(size_t index) const { return this->inputs_[index]; };


} // namespace data_flow
} // namespace sdfg
