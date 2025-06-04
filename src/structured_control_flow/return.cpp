#include "sdfg/structured_control_flow/return.h"

namespace sdfg {
namespace structured_control_flow {

Return::Return(const DebugInfo& debug_info)
    : ControlFlowNode(debug_info) {

      };

void Return::replace(const symbolic::Expression& old_expression,
                     const symbolic::Expression& new_expression) {

};

}  // namespace structured_control_flow
}  // namespace sdfg
