#include "sdfg/structured_control_flow/control_flow_node.h"

namespace sdfg {
namespace structured_control_flow {

ControlFlowNode::ControlFlowNode(size_t element_id, const DebugInfo& debug_info)
    : Element(element_id, debug_info) {

      };

std::string ControlFlowNode::name() const {
    std::string name = std::to_string(this->element_id());
    return "__node_" + name;
};

}  // namespace structured_control_flow
}  // namespace sdfg
