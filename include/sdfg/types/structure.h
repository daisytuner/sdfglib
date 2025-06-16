#pragma once

#include <cassert>
#include <string>
#include <vector>

#include "sdfg/types/type.h"

namespace sdfg {

namespace types {

class Structure : public IType {
   private:
    std::string name_;

   public:
    Structure(const std::string& name);

    Structure(StorageType storage_type, size_t alignment, const std::string& initializer,
              const std::string& name);

    virtual PrimitiveType primitive_type() const override;

    virtual TypeID type_id() const override;

    virtual bool is_symbol() const override;

    const std::string& name() const;

    virtual bool operator==(const IType& other) const override;

    virtual std::unique_ptr<IType> clone() const override;

    virtual std::string print() const override;
};

class StructureDefinition {
   private:
    std::string name_;
    bool is_packed_;
    std::vector<std::unique_ptr<IType>> members_;

   public:
    StructureDefinition(const std::string& name, bool is_packed);

    std::unique_ptr<StructureDefinition> clone() const;

    const std::string& name() const;

    bool is_packed() const;

    size_t num_members() const;

    const IType& member_type(symbolic::Integer index) const;

    void add_member(const IType& member_type);
};

}  // namespace types
}  // namespace sdfg
