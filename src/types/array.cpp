#include "sdfg/types/array.h"

namespace sdfg {
namespace types {

Array::Array(const IType& element_type, const symbolic::Expression& num_elements,
             DeviceLocation device_location, uint address_space, const std::string& initializer

             )
    : element_type_(element_type.clone()),
      num_elements_(num_elements),
      device_location_(device_location),
      address_space_(address_space),
      initializer_(initializer) {};

PrimitiveType Array::primitive_type() const { return this->element_type_->primitive_type(); };

bool Array::is_symbol() const { return false; };

const IType& Array::element_type() const { return *this->element_type_; };

const symbolic::Expression& Array::num_elements() const { return this->num_elements_; };

bool Array::operator==(const IType& other) const {
    if (auto array_type = dynamic_cast<const Array*>(&other)) {
        return symbolic::eq(this->num_elements_, array_type->num_elements_) &&
               *(this->element_type_) == *array_type->element_type_;
    } else {
        return false;
    }
};

std::unique_ptr<IType> Array::clone() const {
    return std::make_unique<Array>(*this->element_type_, this->num_elements_,
                                   this->device_location_, this->address_space_,
                                   this->initializer_);
};

uint Array::address_space() const { return this->address_space_; };

DeviceLocation Array::device_location() const { return this->device_location_; };

std::string Array::initializer() const { return this->initializer_; };

}  // namespace types
}  // namespace sdfg
