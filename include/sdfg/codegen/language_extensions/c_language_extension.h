#pragma once

#include <symengine/printers/codegen.h>

#include "sdfg/codegen/language_extension.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace codegen {

class CLanguageExtension : public LanguageExtension {
public:
    CLanguageExtension() : LanguageExtension() {}

    CLanguageExtension(
        const std::vector<std::string>& external_variables,
        const std::string& external_prefix = ""
    ) : LanguageExtension(external_variables, external_prefix) {}

    const std::string language() const override { return "C"; }

    std::string primitive_type(const types::PrimitiveType prim_type) override;

    std::string declaration(
        const std::string& name, const types::IType& type, bool use_initializer = false, bool use_alignment = false
    ) override;

    std::string type_cast(const std::string& name, const types::IType& type) override;

    std::string subset(const Function& function, const types::IType& type, const data_flow::Subset& subset) override;

    std::string expression(const symbolic::Expression expr) override;

    std::string access_node(const data_flow::AccessNode& node) override;

    std::string tasklet(const data_flow::Tasklet& tasklet) override;

    std::string zero(const types::PrimitiveType prim_type) override;
};

class CSymbolicPrinter : public SymEngine::BaseVisitor<CSymbolicPrinter, SymEngine::CodePrinter> {
private:
    const std::unordered_set<std::string>& external_variables_;
    std::string external_prefix_;

public:
    using SymEngine::CodePrinter::apply;
    using SymEngine::CodePrinter::bvisit;
    using SymEngine::CodePrinter::str_;

    // Special values
    void bvisit(const SymEngine::Infty& x);
    void bvisit(const SymEngine::BooleanAtom& x);
    void bvisit(const SymEngine::Symbol& x);

    // Logical expressions
    void bvisit(const SymEngine::And& x);
    void bvisit(const SymEngine::Or& x);
    void bvisit(const SymEngine::Not& x);
    void bvisit(const SymEngine::Equality& x);
    void bvisit(const SymEngine::Unequality& x);

    // Functions
    void bvisit(const SymEngine::Min& x);
    void bvisit(const SymEngine::Max& x);
    void bvisit(const SymEngine::FunctionSymbol& x);

    void _print_pow(
        std::ostringstream& o,
        const SymEngine::RCP<const SymEngine::Basic>& a,
        const SymEngine::RCP<const SymEngine::Basic>& b
    ) override;

    CSymbolicPrinter(const std::unordered_set<std::string>& external_variables, const std::string& external_prefix = "")
        : external_variables_(external_variables), external_prefix_(external_prefix) {}
};

} // namespace codegen
} // namespace sdfg
