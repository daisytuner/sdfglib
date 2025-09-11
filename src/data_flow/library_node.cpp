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
    const bool side_effect,
    const ImplementationType& implementation_type
)
    : CodeNode(element_id, debug_info, vertex, parent, outputs, inputs), code_(code), side_effect_(side_effect),
      implementation_type_(implementation_type) {}

const LibraryNodeCode& LibraryNode::code() const { return this->code_; };

const ImplementationType& LibraryNode::implementation_type() const { return this->implementation_type_; };

ImplementationType& LibraryNode::implementation_type() { return this->implementation_type_; };

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
