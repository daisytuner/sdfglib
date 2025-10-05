#include "sdfg/structured_control_flow/return.h"

#include "sdfg/function.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace structured_control_flow {

Return::Return(size_t element_id, const DebugInfo& debug_info, const std::string& data)
    : ControlFlowNode(element_id, debug_info), data_(data), unreachable_(false), type_(nullptr) {}

Return::Return(size_t element_id, const DebugInfo& debug_info)
    : ControlFlowNode(element_id, debug_info), data_(""), unreachable_(true), type_(nullptr) {}

Return::Return(size_t element_id, const DebugInfo& debug_info, const std::string& data, const types::IType& type)
    : ControlFlowNode(element_id, debug_info), data_(data), unreachable_(false), type_(type.clone()) {}


const std::string& Return::data() const { return data_; }

const types::IType& Return::type() const { return *type_; }

bool Return::unreachable() const { return unreachable_; }

bool Return::is_data() const { return type_ == nullptr && !unreachable_; }

bool Return::is_unreachable() const { return unreachable_; }

bool Return::is_constant() const { return type_ != nullptr && !unreachable_; }

void Return::validate(const Function& function) const {
    if (is_data()) {
        if (data_ != "" && !function.exists(data_)) {
            throw InvalidSDFGException("Return node with data '" + data_ + "' does not correspond to any container");
        }
        if (unreachable_) {
            throw InvalidSDFGException("Return node cannot be both data and unreachable");
        }
        if (type_ != nullptr) {
            throw InvalidSDFGException("Return node with data cannot have a type");
        }
    } else if (is_constant()) {
        if (function.exists(data_)) {
            throw InvalidSDFGException(
                "Return node with constant data '" + data_ + "' cannot correspond to any container"
            );
        }
        if (type_ == nullptr) {
            throw InvalidSDFGException("Return node with constant data must have a type");
        }
    } else if (is_unreachable()) {
        if (!data_.empty()) {
            throw InvalidSDFGException("Unreachable return node cannot have data");
        }
        if (type_ != nullptr) {
            throw InvalidSDFGException("Unreachable return node cannot have a type");
        }
    } else {
        throw InvalidSDFGException("Return node must be either data, constant, or unreachable");
    }
};

void Return::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    if (is_data() && data_ == old_expression->__str__()) {
        data_ = new_expression->__str__();
    }
};

} // namespace structured_control_flow
} // namespace sdfg
