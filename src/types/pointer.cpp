#include "sdfg/types/pointer.h"

namespace sdfg {
namespace types {

Pointer::Pointer(const IType& pointee_type) : pointee_type_(pointee_type.clone()) {};

Pointer::Pointer(StorageType storage_type, size_t alignment, const std::string& initializer, const IType& pointee_type)
    : IType(storage_type, alignment, initializer), pointee_type_(pointee_type.clone()) {};

std::unique_ptr<IType> Pointer::clone() const {
    return std::make_unique<Pointer>(this->storage_type(), this->alignment(), this->initializer(), *this->pointee_type_);
};

TypeID Pointer::type_id() const { return TypeID::Pointer; };

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

std::string Pointer::print() const { return "Pointer(" + this->pointee_type_->print() + ")"; };

} // namespace types
} // namespace sdfg
