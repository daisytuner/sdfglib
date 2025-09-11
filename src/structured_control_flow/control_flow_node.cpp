#include "sdfg/structured_control_flow/control_flow_node.h"

namespace sdfg {
namespace structured_control_flow {

ControlFlowNode::ControlFlowNode(size_t element_id, const DebugInfoRegion& debug_info)
    : Element(element_id, debug_info) {

      };

} // namespace structured_control_flow
} // namespace sdfg
