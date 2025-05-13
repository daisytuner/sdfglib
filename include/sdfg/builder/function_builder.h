#pragma once

#include <utility>

#include "sdfg/function.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace builder {

class FunctionBuilder {
   protected:
    size_t element_counter_;

    virtual Function& function() const = 0;

   public:
    FunctionBuilder();

    /***** Section: Containers *****/

    const types::IType& add_container(const std::string& name, const types::IType& type,
                                      bool is_argument = false, bool is_external = false) const;

    void remove_container(const std::string& name) const;

    void change_type(const std::string& name, const types::IType& type) const;

    types::StructureDefinition& add_structure(const std::string& name, bool is_packed) const;

    void make_array(const std::string& name, const symbolic::Expression& size) const;

    std::string find_new_name(std::string prefix = "tmp_") const;
};

}  // namespace builder
}  // namespace sdfg