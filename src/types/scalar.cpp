#include "sdfg/types/scalar.h"

namespace sdfg {
namespace types {

Scalar::Scalar(PrimitiveType primitive_type, DeviceLocation device_location, uint address_space,
               const std::string& initializer, size_t alignment)
    : primitive_type_(primitive_type),
      device_location_(device_location),
      address_space_(address_space),
      initializer_(initializer),
      alignment_(alignment) {};

PrimitiveType Scalar::primitive_type() const { return this->primitive_type_; };

bool Scalar::is_symbol() const { return types::is_integer(this->primitive_type_); };

bool Scalar::operator==(const IType& other) const {
    if (auto scalar_type = dynamic_cast<const Scalar*>(&other)) {
        return this->primitive_type_ == scalar_type->primitive_type_ &&
               this->alignment_ == scalar_type->alignment_;
    } else {
        return false;
    }
};

Scalar Scalar::as_signed() const {
    return Scalar(types::as_signed(this->primitive_type_), this->device_location_,
                  this->address_space_, this->initializer_, this->alignment_);
};

Scalar Scalar::as_unsigned() const {
    return Scalar(types::as_unsigned(this->primitive_type_), this->device_location_,
                  this->address_space_, this->initializer_, this->alignment_);
};

std::unique_ptr<IType> Scalar::clone() const {
    return std::make_unique<Scalar>(this->primitive_type_, this->device_location_,
                                    this->address_space_, this->initializer_, this->alignment_);
};

uint Scalar::address_space() const { return this->address_space_; };

DeviceLocation Scalar::device_location() const { return this->device_location_; };

std::string Scalar::initializer() const { return this->initializer_; };

size_t Scalar::alignment() const { return this->alignment_; };

std::string Scalar::print() const {
    return "Scalar(" + std::string(primitive_type_to_string(this->primitive_type_)) + ")";
};

}  // namespace types
}  // namespace sdfg
