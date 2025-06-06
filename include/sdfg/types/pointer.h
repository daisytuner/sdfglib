#pragma once

#include "sdfg/types/type.h"

namespace sdfg {

namespace types {

class Pointer : public IType {
   private:
    std::unique_ptr<IType> pointee_type_;

   public:
    Pointer(const IType& pointee_type);

    Pointer(StorageType storage_type, size_t alignment, const std::string& initializer,
            const IType& pointee_type);

    virtual std::unique_ptr<IType> clone() const override;

    virtual PrimitiveType primitive_type() const override;

    virtual bool is_symbol() const override;

    const IType& pointee_type() const;

    virtual bool operator==(const IType& other) const override;

    virtual std::string print() const override;
};
}  // namespace types
}  // namespace sdfg
