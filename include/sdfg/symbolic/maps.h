/**
 * @file maps.h
 * @brief Analysis of symbol evolution through maps
 *
 * This file provides functions for analyzing symbolic maps, as used in memory
 * accesses with respect to induction variables of loops.
 *
 * ## Map Analysis
 *
 * Key operations:
 * - **Monotonicity checking**: Determines if a map is strictly (positive) monotonic
 * - **Intersection checking**: Determines if the integer domains of two maps intersect
 *
 * @see assumptions.h for information about symbol assumptions and maps
 * @see symbolic.h for building symbolic expressions
 */

#pragma once

#include <unordered_map>

#include "sdfg/symbolic/assumptions.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace symbolic {
namespace maps {

/**
 * @brief Checks if an expression is monotonic with respect to a symbol
 * @param expr The expression to check
 * @param sym The symbol to check monotonicity with respect to
 * @param assums Assumptions about symbols including the evolution map
 * @return true if expr is monotonic (always increasing or always decreasing) as sym evolves
 *
 * An expression is monotonic if it consistently increases as the symbol
 * evolves.
 */
bool is_monotonic(const Expression expr, const Symbol sym, const Assumptions& assums);

/**
 * @brief Checks if the integer domain of two maps intersect
 * @param expr1 First multi-dimensional expression (e.g., memory access pattern)
 * @param expr2 Second multi-dimensional expression
 * @param indvar The induction variable that evolves
 * @param assums1 Assumptions for the first expression including evolution maps
 * @param assums2 Assumptions for the second expression including evolution maps
 * @return true if the integer domains of expr1 and expr2 can overlap
 *
 * @code
 * // Check if A[i] and A[j+5] intersect when both evolve 0 to 10
 * auto i = symbolic::symbol("i");
 * auto j = symbolic::symbol("j");
 * MultiExpression expr1 = {i};
 * MultiExpression expr2 = {symbolic::add(j, symbolic::integer(5))};
 *
 * Assumptions assums1, assums2;
 * assums1[i].add_lower_bound(symbolic::zero());
 * assums1[i].add_upper_bound(symbolic::integer(10));
 * assums2[j].add_lower_bound(symbolic::zero());
 * assums2[j].add_upper_bound(symbolic::integer(10));
 *
 * bool overlap = intersects(expr1, expr2, i, assums1, assums2);  // true (e.g., i=7, j=2)
 * @endcode
 */
bool intersects(
    const MultiExpression& expr1,
    const MultiExpression& expr2,
    const Symbol indvar,
    const Assumptions& assums1,
    const Assumptions& assums2
);

} // namespace maps
} // namespace symbolic
} // namespace sdfg
