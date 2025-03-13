#pragma once

#include "sdfg/types/type.h"

namespace sdfg {

namespace types {

class Array : public IType {
   private:
    std::unique_ptr<IType> element_type_;
    symbolic::Expression num_elements_;
    DeviceLocation device_location_;
    uint address_space_;
    std::string initializer_;

   public:
    Array(const IType& element_type, const symbolic::Expression& num_elements,
          DeviceLocation device_location = DeviceLocation::x86, uint address_space = 0,
          const std::string& initializer = "");

    virtual PrimitiveType primitive_type() const override;

    virtual bool is_symbol() const override;

    const IType& element_type() const;

    const symbolic::Expression& num_elements() const;

    virtual bool operator==(const IType& other) const override;

    virtual std::unique_ptr<IType> clone() const override;

    virtual DeviceLocation device_location() const override;

    virtual uint address_space() const override;

    virtual std::string initializer() const override;
};

}  // namespace types
}  // namespace sdfg
