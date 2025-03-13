#include "sdfg/types/structure.h"

namespace sdfg {
namespace types {

Structure::Structure(const std::string& name, DeviceLocation device_location, uint address_space,
                     const std::string& initializer)
    : name_(name),
      device_location_(device_location),
      address_space_(address_space),
      initializer_(initializer){};

PrimitiveType Structure::primitive_type() const { return PrimitiveType::Void; };

bool Structure::is_symbol() const { return false; };

const std::string& Structure::name() const { return this->name_; };

bool Structure::operator==(const IType& other) const {
    if (auto structure_ = dynamic_cast<const Structure*>(&other)) {
        return this->name_ == structure_->name_;
    } else {
        return false;
    }
};

std::unique_ptr<IType> Structure::clone() const {
    return std::make_unique<Structure>(this->name_, this->device_location_, this->address_space_,
                                       this->initializer_);
};

uint Structure::address_space() const { return this->address_space_; };

DeviceLocation Structure::device_location() const { return this->device_location_; };

std::string Structure::initializer() const { return this->initializer_; };

StructureDefinition::StructureDefinition(const std::string& name) : name_(name), members_(){};

std::unique_ptr<StructureDefinition> StructureDefinition::clone() const {
    auto def = std::make_unique<StructureDefinition>(this->name_);
    for (unsigned i = 0; i < this->num_members(); i++) {
        def->add_member(this->member_type(symbolic::integer(i)));
    }
    return def;
};

const std::string& StructureDefinition::name() const { return this->name_; };

size_t StructureDefinition::num_members() const { return this->members_.size(); };

const IType& StructureDefinition::member_type(symbolic::Integer index) const {
    return *this->members_.at(index->as_uint());
};

void StructureDefinition::add_member(const IType& member_type) {
    this->members_.push_back(member_type.clone());
};

}  // namespace types
}  // namespace sdfg
