#pragma once

#include "sdfg/types/type.h"

namespace sdfg {

namespace types {

class Array : public IType {
   private:
    std::unique_ptr<IType> element_type_;
    symbolic::Expression num_elements_;

   public:
    Array(const IType& element_type, const symbolic::Expression& num_elements);

    Array(StorageType storage_type, size_t alignment, const std::string& initializer,
          const IType& element_type, const symbolic::Expression& num_elements);

    virtual PrimitiveType primitive_type() const override;

    virtual TypeID type_id() const override;

    virtual bool is_symbol() const override;

    const IType& element_type() const;

    const symbolic::Expression& num_elements() const;

    virtual bool operator==(const IType& other) const override;

    virtual std::unique_ptr<IType> clone() const override;

    virtual std::string print() const override;
};

}  // namespace types
}  // namespace sdfg
