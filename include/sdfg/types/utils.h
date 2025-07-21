#pragma once

#include <cassert>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "sdfg/function.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/array.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/structure.h"
#include "sdfg/types/type.h"

namespace sdfg {

namespace data_flow {

typedef std::vector<symbolic::Expression> Subset;

}

namespace types {

const types::IType& infer_type(const sdfg::Function& function, const types::IType& type, const data_flow::Subset& subset);

std::unique_ptr<types::IType> recombine_array_type(const types::IType& type, uint depth, const types::IType& inner_type);

inline constexpr int PEEL_TO_INNERMOST_ELEMENT_FOLLOW_ONLY_OUTER_PTR = -100;
/// Returns the innermost element type of the given type. (for multi-dimension arrays)
/// @param follow_ptr controls how to handle pointers.
///     * Defaults to only following an outermost pointer (as that would be array-like)
///     * If positive count, will allow following that many pointers.
///     * -1 to allow infinite following of pointers.
const IType& peel_to_innermost_element(const IType& type, int follow_ptr = PEEL_TO_INNERMOST_ELEMENT_FOLLOW_ONLY_OUTER_PTR);

/// Returns the size in bytes of one element (of an array) according to ::peel_to_innermost_contiguous_element.
///
/// Otherwise see ::get_type_size.
symbolic::Expression get_contiguous_element_size(const types::IType& type, bool allow_comp_time_eval = true);

/// Returns the size of the given type in bytes , if possible
///
/// @param allow_comp_time_eval whether to emit expressions containing symbolic sizeof to be resolved by the compiler,
///     if the size cannot be determined statically
/// @return Empty RCP if the size is unknown
symbolic::Expression get_type_size(const types::IType& type, bool allow_comp_time_eval = true);

} // namespace types
} // namespace sdfg
