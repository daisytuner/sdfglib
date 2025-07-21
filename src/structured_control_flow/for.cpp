#include "sdfg/structured_control_flow/for.h"

#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace structured_control_flow {

For::
    For(size_t element_id,
        const DebugInfo& debug_info,
        symbolic::Symbol indvar,
        symbolic::Expression init,
        symbolic::Expression update,
        symbolic::Condition condition)
    : StructuredLoop(element_id, debug_info, indvar, init, update, condition) {};

void For::validate() const { this->root_->validate(); };

} // namespace structured_control_flow
} // namespace sdfg
