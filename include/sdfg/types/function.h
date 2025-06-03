#pragma once

#include "sdfg/types/type.h"

namespace sdfg {

namespace types {

class Function : public IType {
   private:
    std::vector<std::unique_ptr<IType>> params_;
    std::unique_ptr<IType> return_type_;
    bool is_var_arg_;
    DeviceLocation device_location_;
    uint address_space_;
    std::string initializer_;

   public:
    Function(const IType& return_type, bool is_var_arg = false,
             DeviceLocation device_location = DeviceLocation::x86, uint address_space = 0,
             const std::string& initializer = "");

    virtual PrimitiveType primitive_type() const override;

    virtual bool is_symbol() const override;

    size_t num_params() const;

    const IType& param_type(symbolic::Integer index) const;

    void add_param(const IType& param);

    const IType& return_type() const;

    bool is_var_arg() const;

    virtual bool operator==(const IType& other) const override;

    virtual std::unique_ptr<IType> clone() const override;

    virtual DeviceLocation device_location() const override;

    virtual uint address_space() const override;

    virtual std::string initializer() const override;

    virtual std::string print() const override;
};

}  // namespace types
}  // namespace sdfg
