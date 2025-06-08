#pragma once

#include <symengine/printers/codegen.h>

#include "sdfg/codegen/language_extension.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace codegen {

class CUDALanguageExtension : public LanguageExtension {
   public:
    std::string primitive_type(const types::PrimitiveType prim_type) override;

    std::string declaration(const std::string& name, const types::IType& type,
                            bool use_initializer = false, bool use_alignment = false) override;

    std::string type_cast(const std::string& name, const types::IType& type) override;

    std::string subset(const Function& function, const types::IType& type,
                       const data_flow::Subset& subset) override;

    std::string expression(const symbolic::Expression& expr) override;

    std::string tasklet(const data_flow::Tasklet& tasklet) override;
};

}  // namespace codegen
}  // namespace sdfg