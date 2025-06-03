#include "sdfg/types/pointer.h"

namespace sdfg {
namespace types {

Pointer::Pointer(const IType& pointee_type, DeviceLocation device_location, uint address_space,
                 const std::string& initializer)
    : pointee_type_(pointee_type.clone()),
      device_location_(device_location),
      address_space_(address_space),
      initializer_(initializer) {};

std::unique_ptr<IType> Pointer::clone() const {
    return std::make_unique<Pointer>(*this->pointee_type_, this->device_location_,
                                     this->address_space_, this->initializer_);
};

PrimitiveType Pointer::primitive_type() const { return this->pointee_type_->primitive_type(); };

bool Pointer::is_symbol() const { return true; };

const IType& Pointer::pointee_type() const { return *this->pointee_type_; };

bool Pointer::operator==(const IType& other) const {
    if (auto pointer_type = dynamic_cast<const Pointer*>(&other)) {
        return *(this->pointee_type_) == *pointer_type->pointee_type_;
    } else {
        return false;
    }
};

uint Pointer::address_space() const { return this->address_space_; };

DeviceLocation Pointer::device_location() const { return this->device_location_; };

std::string Pointer::initializer() const { return this->initializer_; }

std::string Pointer::print() const { return "Pointer(" + this->pointee_type_->print() + ")"; };

}  // namespace types
}  // namespace sdfg
