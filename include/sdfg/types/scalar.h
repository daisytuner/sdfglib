#pragma once

#include "sdfg/types/type.h"

namespace sdfg {

namespace types {

class Scalar : public IType {
   private:
    PrimitiveType primitive_type_;
    DeviceLocation device_location_;
    uint address_space_;
    std::string initializer_;
    size_t alignment_;

   public:
    Scalar(PrimitiveType primitive_type, DeviceLocation device_location = DeviceLocation::x86,
           uint address_space = 0, const std::string& initializer = "", size_t alignment = 0);

    virtual PrimitiveType primitive_type() const override;

    virtual bool is_symbol() const override;

    virtual bool operator==(const IType& other) const override;

    Scalar as_signed() const;

    Scalar as_unsigned() const;

    virtual std::unique_ptr<IType> clone() const override;

    virtual DeviceLocation device_location() const override;

    virtual uint address_space() const override;

    virtual std::string initializer() const override;

    virtual size_t alignment() const override;

    virtual std::string print() const override;
};
}  // namespace types
}  // namespace sdfg
