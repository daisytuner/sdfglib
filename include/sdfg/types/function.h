#pragma once

#include "sdfg/types/type.h"

namespace sdfg {

namespace types {

class Function : public IType {
   private:
    std::vector<std::unique_ptr<IType>> params_;
    std::unique_ptr<IType> return_type_;
    bool is_var_arg_;

   public:
    Function(const IType& return_type, bool is_var_arg = false);

    Function(StorageType storage_type, size_t alignment, const std::string& initializer,
             const IType& return_type, bool is_var_arg = false);

    virtual PrimitiveType primitive_type() const override;

    virtual TypeID type_id() const override;

    virtual bool is_symbol() const override;

    size_t num_params() const;

    const IType& param_type(symbolic::Integer index) const;

    void add_param(const IType& param);

    const IType& return_type() const;

    bool is_var_arg() const;

    virtual bool operator==(const IType& other) const override;

    virtual std::unique_ptr<IType> clone() const override;

    virtual std::string print() const override;
};

}  // namespace types
}  // namespace sdfg
