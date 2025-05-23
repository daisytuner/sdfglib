#include "sdfg/structured_control_flow/structured_loop.h"

namespace sdfg {
namespace structured_control_flow {

StructuredLoop::StructuredLoop(size_t element_id, const DebugInfo& debug_info)
    : ControlFlowNode(element_id, debug_info) {}

}  // namespace structured_control_flow
}  // namespace sdfg
