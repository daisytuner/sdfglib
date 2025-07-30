#pragma once

#include <optional>

#include "sdfg/types/type.h"

namespace sdfg {

namespace types {

class Pointer : public IType {
private:
    std::optional<std::unique_ptr<IType>> pointee_type_;

public:
    Pointer();

    /**
     * WARNING: This is less specific than the COPY-constructor, which still EXISTS and behaves differently!
     *
     * @param pointee_type The type of the object pointed to by this pointer.
     */
    Pointer(const IType& pointee_type);

    Pointer(StorageType storage_type, size_t alignment, const std::string& initializer);

    Pointer(StorageType storage_type, size_t alignment, const std::string& initializer, const IType& pointee_type);

    virtual std::unique_ptr<IType> clone() const override;

    virtual TypeID type_id() const override;

    virtual PrimitiveType primitive_type() const override;

    virtual bool is_symbol() const override;

    bool has_pointee_type() const;

    const IType& pointee_type() const;

    virtual bool operator==(const IType& other) const override;

    virtual std::string print() const override;
};
} // namespace types
} // namespace sdfg
