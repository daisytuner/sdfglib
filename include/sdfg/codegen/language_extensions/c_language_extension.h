#pragma once

#include <symengine/printers/codegen.h>

#include "sdfg/codegen/language_extension.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace codegen {

class CLanguageExtension : public LanguageExtension {
   public:
    std::string primitive_type(const types::PrimitiveType prim_type) override;

    std::string declaration(const std::string& name, const types::IType& type,
                            bool use_initializer = false) override;

    std::string allocation(const std::string& name, const types::IType& type) override;

    std::string deallocation(const std::string& name, const types::IType& type) override;

    std::string type_cast(const std::string& name, const types::IType& type) override;

    std::string subset(const Function& function, const types::IType& type,
                       const data_flow::Subset& subset) override;

    std::string expression(const symbolic::Expression& expr) override;

    std::string tasklet(const data_flow::Tasklet& tasklet) override;

    std::string library_node(const data_flow::LibraryNode& libnode) override;
};

class CSymbolicPrinter : public SymEngine::BaseVisitor<CSymbolicPrinter, SymEngine::CodePrinter> {
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

    void _print_pow(std::ostringstream &o, const SymEngine::RCP<const SymEngine::Basic> &a,
                    const SymEngine::RCP<const SymEngine::Basic> &b) override;
};

}  // namespace codegen
}  // namespace sdfg
