#pragma once

#include <string>

#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/function.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/array.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/structure.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace codegen {

class LanguageExtension {
   public:
    virtual ~LanguageExtension() = default;

    virtual std::string primitive_type(const types::PrimitiveType prim_type) = 0;

    virtual std::string declaration(const std::string& name, const types::IType& type,
                                    bool use_initializer = false, bool use_alignment = false) = 0;

    virtual std::string type_cast(const std::string& name, const types::IType& type) = 0;

    virtual std::string subset(const Function& function, const types::IType& type,
                               const data_flow::Subset& subset) = 0;

    virtual std::string expression(const symbolic::Expression& expr) = 0;

    virtual std::string tasklet(const data_flow::Tasklet& tasklet) = 0;

    virtual std::string library_node(const data_flow::LibraryNode& libnode) = 0;
};

}  // namespace codegen
}  // namespace sdfg
