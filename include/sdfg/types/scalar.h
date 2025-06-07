#pragma once

#include "sdfg/types/type.h"

namespace sdfg {

namespace types {

class Scalar : public IType {
   private:
    PrimitiveType primitive_type_;

   public:
    Scalar(PrimitiveType primitive_type);

    Scalar(StorageType storage_type, size_t alignment, const std::string& initializer,
           PrimitiveType primitive_type);

    virtual PrimitiveType primitive_type() const override;

    virtual bool is_symbol() const override;

    Scalar as_signed() const;

    Scalar as_unsigned() const;

    virtual bool operator==(const IType& other) const override;

    virtual std::unique_ptr<IType> clone() const override;

    virtual std::string print() const override;
};
}  // namespace types
}  // namespace sdfg
