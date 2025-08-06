#include "sdfg/types/utils.h"

#include "sdfg/codegen/utils.h"
#include "sdfg/symbolic/symbolic.h"

#include "sdfg/codegen/language_extensions/c_language_extension.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace types {

const types::IType& infer_type_internal(
    const sdfg::Function& function,
    const types::IType& type,
    const data_flow::Subset& subset,
    bool ignore_contiguous_memory
) {
    if (subset.empty()) {
        return type;
    }

    if (type.type_id() == TypeID::Scalar) {
        if (!subset.empty()) {
            throw InvalidSDFGException("Scalar type must have no subset");
        }

        return type;
    } else if (type.type_id() == TypeID::Array) {
        auto& array_type = static_cast<const types::Array&>(type);

        data_flow::Subset element_subset(subset.begin() + 1, subset.end());
        return infer_type_internal(function, array_type.element_type(), element_subset, ignore_contiguous_memory);
    } else if (type.type_id() == TypeID::Structure) {
        auto& structure_type = static_cast<const types::Structure&>(type);

        auto& definition = function.structure(structure_type.name());

        data_flow::Subset element_subset(subset.begin() + 1, subset.end());
        auto member = SymEngine::rcp_dynamic_cast<const SymEngine::Integer>(subset.at(0));
        return infer_type_internal(function, definition.member_type(member), element_subset, ignore_contiguous_memory);
    } else if (type.type_id() == TypeID::Pointer && ignore_contiguous_memory) {
        auto& pointer_type = static_cast<const types::Pointer&>(type);
        auto& pointee_type = pointer_type.pointee_type();

        data_flow::Subset element_subset(subset.begin() + 1, subset.end());
        return infer_type_internal(function, pointee_type, element_subset, ignore_contiguous_memory);
    } else if (type.type_id() == TypeID::Pointer) {
        throw InvalidSDFGException("Subset references non-contiguous memory");
    }

    throw InvalidSDFGException("Type inference failed because of unknown type");
};

const types::IType& infer_type(
    const sdfg::Function& function,
    const types::IType& type,
    const data_flow::Subset& subset,
    bool ignore_contiguous_memory
) {
    if (subset.empty()) {
        return type;
    }

    if (type.type_id() == TypeID::Pointer) {
        auto& pointer_type = static_cast<const types::Pointer&>(type);
        auto& pointee_type = pointer_type.pointee_type();

        data_flow::Subset element_subset(subset.begin() + 1, subset.end());
        return infer_type_internal(function, pointee_type, element_subset, ignore_contiguous_memory);
    } else {
        return infer_type_internal(function, type, subset, ignore_contiguous_memory);
    }
};

std::unique_ptr<types::IType> recombine_array_type(const types::IType& type, uint depth, const types::IType& inner_type) {
    if (depth == 0) {
        return inner_type.clone();
    } else {
        if (auto atype = dynamic_cast<const types::Array*>(&type)) {
            return std::make_unique<types::Array>(
                atype->storage_type(),
                atype->alignment(),
                atype->initializer(),
                *recombine_array_type(atype->element_type(), depth - 1, inner_type).get(),
                atype->num_elements()
            );
        } else {
            throw std::runtime_error("construct_type: Non array types are not supported yet!");
        }
    }
};

const IType& peel_to_innermost_element(const IType& type, int follow_ptr) {
    int next_follow = follow_ptr;
    if (follow_ptr == PEEL_TO_INNERMOST_ELEMENT_FOLLOW_ONLY_OUTER_PTR) {
        next_follow = 0; // only follow an outermost pointer
    }

    switch (type.type_id()) {
        case TypeID::Array:
            return peel_to_innermost_element(dynamic_cast<const types::Array&>(type).element_type(), next_follow);
        case TypeID::Reference:
            return peel_to_innermost_element(dynamic_cast<const codegen::Reference&>(type).reference_type(), next_follow);
        case TypeID::Pointer:
            if (follow_ptr != 0) {
                if (follow_ptr != PEEL_TO_INNERMOST_ELEMENT_FOLLOW_ONLY_OUTER_PTR) {
                    next_follow = follow_ptr - 1; // follow one less pointer
                }

                return peel_to_innermost_element(dynamic_cast<const types::Pointer&>(type).pointee_type(), next_follow);
            }
            // fall back to cut-off if we did not follow the pointer
        default:
            return type;
    }
}

symbolic::Expression get_contiguous_element_size(const types::IType& type, bool allow_comp_time_eval) {
    // need to peel explicitly, primitive_type() would follow ALL pointers, even ***, even though this is not contiguous
    auto& innermost = peel_to_innermost_element(type, PEEL_TO_INNERMOST_ELEMENT_FOLLOW_ONLY_OUTER_PTR);

    return get_type_size(innermost, allow_comp_time_eval);
}

symbolic::Expression get_type_size(const types::IType& type, bool allow_comp_time_eval) {
    bool only_symbolic = false;

    auto id = type.type_id();
    if (id == TypeID::Pointer || id == TypeID::Reference || id == TypeID::Function) {
        // TODO NEED target info to know pointer size (4 or 8 bytes?) !!
        only_symbolic = true;
    } else if (id == TypeID::Structure) {
        // TODO if we have the target definition, we could evaluate the StructureDefinition to a size
        only_symbolic = true;
    } else if (id == TypeID::Array) {
        auto& arr = dynamic_cast<const types::Array&>(type);
        auto inner_element_size = get_type_size(arr.element_type(), allow_comp_time_eval);
        if (!inner_element_size.is_null()) {
            return symbolic::mul(inner_element_size, arr.num_elements());
        } else {
            return {};
        }
    }

    if (only_symbolic) {
        // Could not statically figure out the size
        // Could be struct we could evaluate by its definition or sth. we do not understand here
        if (allow_comp_time_eval) {
            return symbolic::size_of_type(type);
        } else { // size unknown
            return {};
        }
    } else { // should just be a primitive type
        auto prim_type = type.primitive_type();

        long size_of_type = static_cast<long>(types::bit_width(prim_type)) / 8;
        if (size_of_type != 0) {
            return symbolic::integer(size_of_type);
        } else {
            codegen::CLanguageExtension lang;
            std::cerr << "Unexpected primitive_type " << primitive_type_to_string(prim_type) << " of "
                      << lang.declaration("", type) << ", unknown size";
            return {};
        }
    }
}

} // namespace types
} // namespace sdfg
