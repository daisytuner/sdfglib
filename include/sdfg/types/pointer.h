#pragma once

#include "sdfg/types/type.h"

namespace sdfg {

namespace types {

class Pointer : public IType {
   private:
    std::unique_ptr<IType> pointee_type_;
    DeviceLocation device_location_;
    uint address_space_;
    std::string initializer_;
    size_t alignment_;

   public:
    Pointer(const IType& pointee_type, DeviceLocation device_location = DeviceLocation::x86,
            uint address_space = 0, const std::string& initializer = "", size_t alignment = 0);

    virtual std::unique_ptr<IType> clone() const override;

    virtual PrimitiveType primitive_type() const override;

    virtual bool is_symbol() const override;

    const IType& pointee_type() const;

    virtual bool operator==(const IType& other) const override;

    virtual DeviceLocation device_location() const override;

    virtual uint address_space() const override;

    virtual std::string initializer() const override;

    virtual size_t alignment() const override;

    virtual std::string print() const override;
};
}  // namespace types
}  // namespace sdfg
