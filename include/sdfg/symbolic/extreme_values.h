/**
 * @file extreme_values.h
 * @brief Computing bounds of symbolic expressions using assumptions
 *
 * This file provides functions for computing the minimum and maximum values that
 * symbolic expressions can take, given a set of assumptions about symbol bounds.
 *
 * ## Bound Computation
 *
 * The bound computation uses assumptions about symbols (from assumptions.h) to determine
 * the extreme values (minimum and maximum) that an expression can reach. The computation
 * considers:
 * - Symbol bounds from assumptions
 * - Parameter symbols (treated as unknowns with their bounds)
 * - Tight vs. loose bounds (exact vs. conservative estimates)
 *
 * ## Example Usage
 *
 * @code
 * // Compute bounds for expression 2*i + j where 0 <= i < 10 and 0 <= j < 5
 * auto i = symbolic::symbol("i");
 * auto j = symbolic::symbol("j");
 * auto expr = symbolic::add(symbolic::mul(symbolic::integer(2), i), j);
 *
 * Assumptions assums;
 * assums[i].add_lower_bound(symbolic::zero());
 * assums[i].add_upper_bound(symbolic::integer(10));
 * assums[j].add_lower_bound(symbolic::zero());
 * assums[j].add_upper_bound(symbolic::integer(5));
 *
 * SymbolSet params;  // Empty - treat both as iteration variables
 * auto min_val = minimum(expr, params, assums);  // Result: 0
 * auto max_val = maximum(expr, params, assums);  // Result: 24 (2*9 + 4)
 * @endcode
 *
 * @see assumptions.h for information about symbol assumptions
 * @see symbolic.h for building symbolic expressions
 */

#pragma once

#include "sdfg/symbolic/assumptions.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace symbolic {

/**
 * @brief Compute the minimum of an expression
 *
 * @param expr The expression to compute the minimum of
 * @param parameters A set of symbols to treat as parameters (with unknown but bounded values)
 * @param assumptions A set of assumptions about bounds of symbols
 * @return The minimum of the expression, or null if the expression is not bounded
 *
 * Computes the minimum value that the expression can take given the assumptions.
 * Symbols in the parameters set are treated as unknowns whose specific values affect
 * the result, while other symbols are treated as iteration variables that can be
 * optimized over to find the minimum.
 *
 * @note This is the legacy version. Consider using minimum_new() for improved analysis.
 */
Expression minimum(const Expression expr, const SymbolSet& parameters, const Assumptions& assumptions);

/**
 * @brief Compute the maximum of an expression
 *
 * @param expr The expression to compute the maximum of
 * @param parameters A set of symbols to treat as parameters (with unknown but bounded values)
 * @param assumptions A set of assumptions about bounds of symbols
 * @return The maximum of the expression, or null if the expression is not bounded
 *
 * Computes the maximum value that the expression can take given the assumptions.
 * Symbols in the parameters set are treated as unknowns whose specific values affect
 * the result, while other symbols are treated as iteration variables that can be
 * optimized over to find the maximum.
 *
 * @note This is the legacy version. Consider using maximum_new() for improved analysis.
 */
Expression maximum(const Expression expr, const SymbolSet& parameters, const Assumptions& assumptions);

/**
 * @brief Compute the minimum of an expression with tight/loose bound selection
 *
 * @param expr The expression to compute the minimum of
 * @param parameters A set of symbols to treat as parameters
 * @param assumptions A set of assumptions about bounds of symbols
 * @param tight If true, compute tight (exact) bounds; if false, compute loose (conservative) bounds
 * @return The minimum of the expression, or null if the expression is not bounded
 *
 * This is an improved version of minimum() that supports computing either tight (exact)
 * or loose (conservative) bounds. Tight bounds provide exact minimum values but may be
 * more computationally expensive, while loose bounds provide safe under-approximations.
 */
Expression minimum_new(const Expression expr, const SymbolSet& parameters, const Assumptions& assumptions, bool tight);

/**
 * @brief Compute the maximum of an expression with tight/loose bound selection
 *
 * @param expr The expression to compute the maximum of
 * @param parameters A set of symbols to treat as parameters
 * @param assumptions A set of assumptions about bounds of symbols
 * @param tight If true, compute tight (exact) bounds; if false, compute loose (conservative) bounds
 * @return The maximum of the expression, or null if the expression is not bounded
 *
 * This is an improved version of maximum() that supports computing either tight (exact)
 * or loose (conservative) bounds. Tight bounds provide exact maximum values but may be
 * more computationally expensive, while loose bounds provide safe over-approximations.
 */
Expression maximum_new(const Expression expr, const SymbolSet& parameters, const Assumptions& assumptions, bool tight);

} // namespace symbolic
} // namespace sdfg
