#pragma once

#include <symengine/printers/codegen.h>

#include <string>

#include "sdfg/codegen/language_extension.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace codegen {

class CUDALanguageExtension : public LanguageExtension {
public:
    CUDALanguageExtension() : LanguageExtension() {}

    CUDALanguageExtension(const std::vector<std::string>& external_variables) : LanguageExtension(external_variables) {}

    std::string primitive_type(const types::PrimitiveType prim_type) override;

    std::string declaration(
        const std::string& name, const types::IType& type, bool use_initializer = false, bool use_alignment = false
    ) override;

    std::string type_cast(const std::string& name, const types::IType& type) override;

    std::string subset(const Function& function, const types::IType& type, const data_flow::Subset& subset) override;

    std::string expression(const symbolic::Expression& expr) override;

    std::string access_node(const data_flow::AccessNode& node) override;

    std::string tasklet(const data_flow::Tasklet& tasklet) override;

    std::string zero(const types::PrimitiveType prim_type) override;
};

} // namespace codegen
} // namespace sdfg
