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
#include "sdfg/exceptions.h"
#include "sdfg/helpers/helpers.h"
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

typedef StringEnum FunctionType;
inline FunctionType FunctionType_CPU{"CPU"};
inline FunctionType FunctionType_NV_GLOBAL{"NV_GLOBAL"};

enum LinkageType { LinkageType_External, LinkageType_Internal };

class Function {
    friend class sdfg::builder::FunctionBuilder;

protected:
    size_t element_counter_;

    // Name
    std::string name_;
    FunctionType type_;

    // Data definition
    std::unordered_map<std::string, std::unique_ptr<types::IType>> containers_;
    std::unordered_map<std::string, std::unique_ptr<types::StructureDefinition>> structures_;

    // External data
    std::vector<std::string> arguments_;
    std::vector<std::string> externals_;
    std::unordered_map<std::string, LinkageType> externals_linkage_types_;

    // Symbolic assumptions
    symbolic::Assumptions assumptions_;

    // Metadata
    std::unordered_map<std::string, std::string> metadata_;

    // Debug information
    DebugTable debug_info_;

    Function(const std::string& name, FunctionType type);

public:
    Function(const Function& function) = delete;

    virtual ~Function() = default;

    // Static types for reserved symbols
    static const std::unique_ptr<types::Scalar> NVPTX_SYMBOL_TYPE;
    static const std::unique_ptr<types::Pointer> CONST_POINTER_TYPE;

    /***** Section: Definition *****/

    const std::string& name() const;

    std::string& name();

    FunctionType type() const;

    size_t element_counter() const;

    /**
     * Validates the function.
     *
     * @throw InvalidSDFGException if the function is invalid
     */
    virtual void validate() const;

    bool exists(const std::string& name) const;

    auto containers() const { return std::views::keys(this->containers_); };

    const types::IType& type(const std::string& name) const;

    auto structures() const { return std::views::keys(this->structures_); };

    const types::StructureDefinition& structure(const std::string& name) const;

    const std::vector<std::string>& arguments() const;

    const std::vector<std::string>& externals() const;

    bool is_argument(const std::string& name) const;

    bool is_external(const std::string& name) const;

    bool is_transient(const std::string& name) const;

    LinkageType linkage_type(const std::string& name) const;

    symbolic::SymbolSet parameters() const;

    bool has_assumption(const symbolic::Symbol& symbol) const;

    const symbolic::Assumption& assumption(const symbolic::Symbol& symbol) const;

    symbolic::Assumption& assumption(const symbolic::Symbol& symbol);

    const symbolic::Assumptions& assumptions() const;

    void add_metadata(const std::string& key, const std::string& value);

    void remove_metadata(const std::string& key);

    const std::string& metadata(const std::string& key) const;

    const std::unordered_map<std::string, std::string>& metadata() const;

    /***** Section: Debug Info *****/
    DebugTable& debug_info() { return this->debug_info_; };
    const DebugTable& debug_info() const { return this->debug_info_; };

    void debug_info(const DebugTable& debug_info) { this->debug_info_ = debug_info; };
};
} // namespace sdfg
