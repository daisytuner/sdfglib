#pragma once

#include <cassert>
#include <fstream>
#include <functional>
#include <list>
#include <memory>
#include <nlohmann/json.hpp>
#include <ranges>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "sdfg/element.h"
#include "sdfg/helpers/helpers.h"
#include "sdfg/symbolic/analysis.h"
#include "sdfg/symbolic/assumptions.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/array.h"
#include "sdfg/types/function.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/structure.h"
#include "sdfg/types/type.h"

using json = nlohmann::json;

namespace sdfg {

namespace builder {
class FunctionBuilder;
}

static std::string external_suffix = "__daisy__internal__";

class Function {
    friend class sdfg::builder::FunctionBuilder;

   protected:
    // Name
    std::string name_;

    // Data definition
    std::unordered_map<std::string, std::unique_ptr<types::IType>> containers_;
    std::unordered_map<std::string, std::unique_ptr<types::StructureDefinition>> structures_;

    // External data
    std::vector<std::string> arguments_;
    std::vector<std::string> externals_;

    // Symbolic assumptions
    symbolic::Assumptions assumptions_;

    Function(const std::string& name);

   public:
    Function(const Function& function) = delete;

    /***** Section: Definition *****/

    std::string name() const;

    virtual const DebugInfo debug_info() const = 0;

    bool exists(const std::string& name) const;

    auto containers() const { return std::views::keys(this->containers_); };

    const types::IType& type(const std::string& name) const;

    auto structures() const { return std::views::keys(this->structures_); };

    const types::StructureDefinition& structure(const std::string& name) const;

    const std::vector<std::string>& arguments() const;

    const std::vector<std::string>& externals() const;

    bool is_argument(const std::string& name) const;

    bool is_external(const std::string& name) const;

    bool is_internal(const std::string& name) const;

    bool is_transient(const std::string& name) const;

    bool has_assumption(const symbolic::Symbol& symbol) const;

    const symbolic::Assumption& assumption(const symbolic::Symbol& symbol) const;

    symbolic::Assumption& assumption(const symbolic::Symbol& symbol);

    const symbolic::Assumptions& assumptions() const;

    /***** Section: Serialization *****/
};
}  // namespace sdfg