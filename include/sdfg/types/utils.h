#pragma once

#include <cassert>
#include <memory>
#include <string>
#include <vector>

#include "sdfg/function.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/type.h"
#include "sdfg/analysis/analysis.h"

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

/// Infers the type of a container from its users.
/// @param analysis_manager Analysis manager to use for finding users.
/// @param sdfg The SDFG to use for type inference.
/// @param container The name of the container to infer the type for.
/// @return The inferred type of the container.
std::unique_ptr<typename types::IType> infer_type_from_container(analysis::AnalysisManager& analysis_manager, const StructuredSDFG& sdfg, std::string container);

} // namespace types
} // namespace sdfg
