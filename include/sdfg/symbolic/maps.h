/**
 * @file maps.h
 * @brief Analysis of symbol evolution through maps
 *
 * This file provides functions for analyzing how symbols evolve through maps,
 * particularly for loop induction variables. Maps describe how symbol values
 * change from one iteration to the next, enabling analysis of:
 * - Monotonicity of expressions
 * - Intersection of iteration spaces
 * - Memory access patterns in loops
 *
 * ## Map Analysis
 *
 * In the symbolic system, a map describes how a symbol evolves. For example,
 * a loop counter i with map(i) = i + 1 evolves by incrementing by 1 each iteration.
 * These maps are stored in assumptions and are used to analyze loop behavior.
 *
 * Key operations:
 * - **Monotonicity checking**: Determines if an expression increases/decreases monotonically
 * - **Intersection checking**: Determines if two iteration spaces overlap
 *
 * ## Example Usage
 *
 * @code
 * // Check if i is monotonic in a loop where i evolves as i' = i + 1
 * auto i = symbolic::symbol("i");
 * auto expr = symbolic::mul(symbolic::integer(2), i);  // 2*i
 * 
 * Assumptions assums;
 * assums[i].add_lower_bound(symbolic::zero());
 * assums[i].add_upper_bound(symbolic::integer(10));
 * assums[i].map(symbolic::add(i, symbolic::one()));  // i' = i + 1
 * 
 * bool mono = is_monotonic(expr, i, assums);  // true (2*i increases as i increases)
 * @endcode
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
 * An expression is monotonic if it consistently increases or decreases as the symbol
 * evolves according to its map in the assumptions. This is useful for determining
 * whether memory accesses follow a predictable pattern in loops.
 */
bool is_monotonic(const Expression expr, const Symbol sym, const Assumptions& assums);

/**
 * @brief Checks if two iteration spaces intersect
 * @param expr1 First multi-dimensional expression (e.g., memory access pattern)
 * @param expr2 Second multi-dimensional expression
 * @param indvar The induction variable that evolves through the iterations
 * @param assums1 Assumptions for the first expression including evolution maps
 * @param assums2 Assumptions for the second expression including evolution maps
 * @return true if the iteration spaces of expr1 and expr2 can overlap
 *
 * Determines whether two iteration spaces (e.g., two memory access patterns in different
 * loops) can produce the same values for any combination of their iteration variables.
 * This is critical for dependence analysis in parallel loops.
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
