#include "sdfg/structured_control_flow/return.h"

namespace sdfg {
namespace structured_control_flow {

Return::Return(size_t element_id, const DebugInfo& debug_info, const std::string& data)
    : ControlFlowNode(element_id, debug_info), data_(data) {}

bool Return::has_data() const { return !data_.empty(); }

const std::string& Return::data() const { return data_; }

void Return::validate(const Function& function) const {};

void Return::replace(const symbolic::Expression& old_expression, const symbolic::Expression& new_expression) {

};

} // namespace structured_control_flow
} // namespace sdfg
