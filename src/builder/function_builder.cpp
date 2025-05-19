#include "sdfg/builder/function_builder.h"

namespace sdfg {
namespace builder {

FunctionBuilder::FunctionBuilder() : element_counter_(1) {};

const types::IType& FunctionBuilder::add_container(const std::string& name,
                                                   const types::IType& type, bool is_argument,
                                                   bool is_external) const {
    if (is_argument && is_external) {
        throw std::invalid_argument("Container cannot be both an argument and an external");
    }

    auto res = this->function().containers_.insert({name, type.clone()});
    assert(res.second);

    if (is_argument) {
        this->function().arguments_.push_back(name);
    }
    if (is_external) {
        this->function().externals_.push_back(name);
    }
    if (type.is_symbol() && dynamic_cast<const types::Scalar*>(&type)) {
        auto sym = symbolic::symbol(name);
        this->function().assumptions_.insert({sym, symbolic::Assumption::create(sym, type)});
    }

    return *(*res.first).second;
};

void FunctionBuilder::remove_container(const std::string& name) const {
    auto& function = this->function();
    if (!function.is_transient(name)) {
        throw std::invalid_argument("Container is not transient");
    }

    auto& type = function.containers_[name];
    if (type->is_symbol() && dynamic_cast<const types::Scalar*>(type.get())) {
        function.assumptions_.erase(symbolic::symbol(name));
    }

    function.containers_.erase(name);
};

void FunctionBuilder::change_type(const std::string& name, const types::IType& type) const {
    auto& function = this->function();
    if (!function.is_transient(name)) {
        throw std::invalid_argument("Container is not transient");
    }

    function.containers_[name] = type.clone();
};

types::StructureDefinition& FunctionBuilder::add_structure(const std::string& name,
                                                           bool is_packed) const {
    auto res = this->function().structures_.insert(
        {name, std::make_unique<types::StructureDefinition>(name, is_packed)});
    assert(res.second);

    return *(*res.first).second;
};

void FunctionBuilder::make_array(const std::string& name, const symbolic::Expression& size) const {
    auto& function = this->function();
    if (!function.is_transient(name)) {
        throw std::invalid_argument("Container is not transient");
    }

    auto& old_type = function.containers_[name];

    if (old_type->is_symbol()) {
        function.assumptions_.erase(symbolic::symbol(name));
    }

    function.containers_[name] = std::make_unique<types::Array>(*old_type, size);
};

std::string FunctionBuilder::find_new_name(std::string prefix) const {
    size_t i = 0;
    std::string new_name = prefix + std::to_string(i);
    while (this->function().exists(new_name)) {
        i++;
        new_name = prefix + std::to_string(i);
    }
    return new_name;
};

}  // namespace builder
}  // namespace sdfg
