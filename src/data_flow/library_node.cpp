#include "sdfg/data_flow/library_node.h"

#include <string>

namespace sdfg {
namespace data_flow {

LibraryNode::LibraryNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    DataFlowGraph& parent,
    const LibraryNodeCode& code,
    const std::vector<std::string>& outputs,
    const std::vector<std::string>& inputs,
    const bool side_effect
)
    : CodeNode(element_id, debug_info, vertex, parent), code_(code), outputs_(outputs), inputs_(inputs),
      side_effect_(side_effect) {

      };

const LibraryNodeCode& LibraryNode::code() const { return this->code_; };

const std::vector<std::string>& LibraryNode::inputs() const { return this->inputs_; };

const std::vector<std::string>& LibraryNode::outputs() const { return this->outputs_; };

const std::string& LibraryNode::input(size_t index) const { return this->inputs_[index]; };

const std::string& LibraryNode::output(size_t index) const { return this->outputs_[index]; };

bool LibraryNode::side_effect() const { return this->side_effect_; };

bool LibraryNode::needs_connector(size_t index) const {
    // Is non-constant, if starts with _in prefix
    if (this->inputs_[index].compare(0, 3, "_in") == 0) {
        return true;
    }
    return false;
};

std::string LibraryNode::toStr() const { return std::string(this->code_.value()); }

} // namespace data_flow
} // namespace sdfg
