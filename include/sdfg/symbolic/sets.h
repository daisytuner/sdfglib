#pragma once

#include "sdfg/symbolic/assumptions.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace symbolic {

/**
 * @brief Interprets the expressions as integer sets and checks if expr1 is a subset of expr2.
 *
 * @param expr1 The first expression to check.
 * @param expr2 The second expression to check.
 * @param assums1 The assumptions for the first expression.
 * @param assums2 The assumptions for the second expression.
 * @return true if expr1 is a subset of expr2, false otherwise.
 */
bool is_subset(
    const MultiExpression& expr1, const MultiExpression& expr2, const Assumptions& assums1, const Assumptions& assums2
);

/**
 * @brief Interprets the expressions as integer sets and checks if expr1 is disjoint from expr2.
 *
 * @param expr1 The first expression to check.
 * @param expr2 The second expression to check.
 * @param assums1 The assumptions for the first expression.
 * @param assums2 The assumptions for the second expression.
 * @return true if expr1 is disjoint from expr2, false otherwise.
 */
bool is_disjoint(
    const MultiExpression& expr1, const MultiExpression& expr2, const Assumptions& assums1, const Assumptions& assums2
);

/** * @brief Checks if two sets of symbols intersect.
 * * @param set1 The first set of symbols.
 * @param set2 The second set of symbols.
 * @return true if the sets intersect, false otherwise.
 */
bool intersects(SymbolSet set1, SymbolSet set2);

} // namespace symbolic
} // namespace sdfg
