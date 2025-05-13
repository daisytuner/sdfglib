#include "sdfg/types/function.h"

namespace sdfg {
namespace types {

Function::Function(const IType& return_type, bool is_var_arg, DeviceLocation device_location,
                   uint address_space, const std::string& initializer)
    : return_type_(return_type.clone()),
      is_var_arg_(is_var_arg),
      device_location_(device_location),
      address_space_(address_space),
      initializer_(initializer) {}

PrimitiveType Function::primitive_type() const { return PrimitiveType::Void; }

bool Function::is_symbol() const { return false; }

size_t Function::num_params() const { return this->params_.size(); }

const IType& Function::param_type(symbolic::Integer index) const {
    return *this->params_[index->as_int()];
}

void Function::add_param(const IType& param) { this->params_.push_back(param.clone()); }

const IType& Function::return_type() const { return *this->return_type_; }

bool Function::is_var_arg() const { return this->is_var_arg_; }

bool Function::operator==(const IType& other) const { return false; }

std::unique_ptr<IType> Function::clone() const {
    auto new_function =
        std::make_unique<Function>(*this->return_type_, this->is_var_arg_, this->device_location_,
                                   this->address_space_, this->initializer_);
    for (const auto& param : this->params_) {
        new_function->add_param(*param);
    }
    return new_function;
}

DeviceLocation Function::device_location() const { return this->device_location_; }

uint Function::address_space() const { return this->address_space_; }

std::string Function::initializer() const { return this->initializer_; }

}  // namespace types
}  // namespace sdfg