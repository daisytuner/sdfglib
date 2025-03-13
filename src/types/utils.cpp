#include "sdfg/types/utils.h"

namespace sdfg {
namespace types {

const types::IType& infer_type(const Function& function, const types::IType& type,
                               const data_flow::Subset& subset) {
    if (subset.empty()) {
        return type;
    }

    if (auto scalar_type = dynamic_cast<const types::Scalar*>(&type)) {
        return *scalar_type;
    } else if (auto array_type = dynamic_cast<const types::Array*>(&type)) {
        data_flow::Subset element_subset(subset.begin() + 1, subset.end());
        return infer_type(function, array_type->element_type(), element_subset);
    } else if (auto pointer_type = dynamic_cast<const types::Pointer*>(&type)) {
        data_flow::Subset element_subset(subset.begin() + 1, subset.end());
        return infer_type(function, pointer_type->pointee_type(), element_subset);
    } else if (auto structure_type = dynamic_cast<const types::Structure*>(&type)) {
        auto& definition = function.structure(structure_type->name());

        data_flow::Subset element_subset(subset.begin() + 1, subset.end());
        auto member = SymEngine::rcp_dynamic_cast<const SymEngine::Integer>(subset.at(0));
        return infer_type(function, definition.member_type(member), element_subset);
    }

    assert(false);
};

std::unique_ptr<types::IType> recombine_array_type(const types::IType& type, uint depth,
                                                   const types::IType& inner_type) {
    if (depth == 0) {
        return inner_type.clone();
    } else {
        if (auto atype = dynamic_cast<const types::Array*>(&type)) {
            return std::make_unique<types::Array>(
                *recombine_array_type(atype->element_type(), depth - 1, inner_type).get(),
                atype->num_elements(), atype->device_location(), atype->address_space(),
                atype->initializer());
        } else {
            throw std::runtime_error("construct_type: Non array types are not supported yet!");
        }
    }
};

}  // namespace types
}  // namespace sdfg
