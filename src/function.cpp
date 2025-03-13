#include "sdfg/function.h"

using json = nlohmann::json;

namespace sdfg {

Function::Function(const std::string& name)
    : name_(name){

      };

std::string Function::name() const { return this->name_; };

bool Function::exists(const std::string& name) const {
    return this->containers_.find(name) != this->containers_.end() ||
           symbolic::is_pointer(symbolic::symbol(name));
};

const types::IType& Function::type(const std::string& name) const {
    auto entry = this->containers_.find(name);
    assert((entry != this->containers_.end()) && "Container does not exist in SDFG");
    return *entry->second;
};

const types::StructureDefinition& Function::structure(const std::string& name) const {
    auto entry = this->structures_.find(name);
    assert((entry != this->structures_.end()) && "Structure does not exist in SDFG");
    return *entry->second;
};

const std::vector<std::string>& Function::arguments() const { return this->arguments_; };

const std::vector<std::string>& Function::externals() const { return this->externals_; };

bool Function::is_argument(const std::string& name) const {
    return std::find(this->arguments_.begin(), this->arguments_.end(), name) !=
           this->arguments_.end();
};

bool Function::is_external(const std::string& name) const {
    return std::find(this->externals_.begin(), this->externals_.end(), name) !=
           this->externals_.end();
};

bool Function::is_internal(const std::string& name) const {
    return helpers::endswith(name, external_suffix) &&
           is_external(name.substr(0, name.length() - external_suffix.length()));
};

bool Function::is_transient(const std::string& name) const {
    return !this->is_argument(name) && !this->is_external(name) && !this->is_internal(name);
};

bool Function::has_assumption(const symbolic::Symbol& symbol) const {
    return this->assumptions_.find(symbol) != this->assumptions_.end();
};

const symbolic::Assumption& Function::assumption(const symbolic::Symbol& symbol) const {
    auto entry = this->assumptions_.find(symbol);
    assert((entry != this->assumptions_.end()) && "Assumptions do not exist in SDFG");
    return entry->second;
};

symbolic::Assumption& Function::assumption(const symbolic::Symbol& symbol) {
    auto entry = this->assumptions_.find(symbol);
    assert((entry != this->assumptions_.end()) && "Assumptions do not exist in SDFG");
    return entry->second;
};

const symbolic::Assumptions& Function::assumptions() const { return this->assumptions_; };

}  // namespace sdfg