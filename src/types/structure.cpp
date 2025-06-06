#include "sdfg/types/structure.h"

namespace sdfg {
namespace types {

Structure::Structure(const std::string& name) : name_(name) {};

Structure::Structure(StorageType storage_type, size_t alignment, const std::string& initializer,
                     const std::string& name)
    : IType(storage_type, alignment, initializer), name_(name) {};

PrimitiveType Structure::primitive_type() const { return PrimitiveType::Void; };

bool Structure::is_symbol() const { return false; };

const std::string& Structure::name() const { return this->name_; };

bool Structure::operator==(const IType& other) const {
    if (auto structure_ = dynamic_cast<const Structure*>(&other)) {
        return this->name_ == structure_->name_ && this->alignment_ == structure_->alignment_;
    } else {
        return false;
    }
};

std::unique_ptr<IType> Structure::clone() const {
    return std::make_unique<Structure>(this->storage_type(), this->alignment(), this->initializer(),
                                       this->name_);
};

std::string Structure::print() const { return "Structure(" + this->name_ + ")"; };

StructureDefinition::StructureDefinition(const std::string& name, bool is_packed)
    : name_(name), is_packed_(is_packed), members_() {};

std::unique_ptr<StructureDefinition> StructureDefinition::clone() const {
    auto def = std::make_unique<StructureDefinition>(this->name_, this->is_packed_);
    for (unsigned i = 0; i < this->num_members(); i++) {
        def->add_member(this->member_type(symbolic::integer(i)));
    }
    return def;
};

const std::string& StructureDefinition::name() const { return this->name_; };

bool StructureDefinition::is_packed() const { return this->is_packed_; };

size_t StructureDefinition::num_members() const { return this->members_.size(); };

const IType& StructureDefinition::member_type(symbolic::Integer index) const {
    return *this->members_.at(index->as_uint());
};

void StructureDefinition::add_member(const IType& member_type) {
    this->members_.push_back(member_type.clone());
};

}  // namespace types
}  // namespace sdfg
