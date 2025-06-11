#include "sdfg/data_flow/thread_barrier_node.h"

#include "sdfg/data_flow/library_node.h"

namespace sdfg {
namespace data_flow {

ThreadBarrierNode::ThreadBarrierNode(const DebugInfo& debug_info, const graph::Vertex vertex,
                                     DataFlowGraph& parent)
    : LibraryNode(debug_info, vertex, parent, LibraryNodeCode{"barrier_local"}, {}, {}, true) {

      };

const LibraryNodeCode& ThreadBarrierNode::code() const { return this->code_; };

const std::vector<std::string>& ThreadBarrierNode::inputs() const { return this->inputs_; };

const std::vector<std::string>& ThreadBarrierNode::outputs() const { return this->outputs_; };

const std::string& ThreadBarrierNode::input(size_t index) const { return this->inputs_[index]; };

const std::string& ThreadBarrierNode::output(size_t index) const { return this->outputs_[index]; };

bool ThreadBarrierNode::side_effect() const { return this->side_effect_; };

bool ThreadBarrierNode::needs_connector(size_t index) const {
    // Is non-constant, if starts with _in prefix
    if (this->inputs_[index].compare(0, 3, "_in") == 0) {
        return true;
    }
    return false;
};

}  // namespace data_flow
}  // namespace sdfg
