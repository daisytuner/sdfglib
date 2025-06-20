#pragma once

#include <string>
#include <vector>

#include "sdfg/symbolic/assumptions.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace symbolic {

typedef std::vector<Expression> MultiExpression;

/**
 * @brief Interprets the expressions as integer sets and checks if expr1 is a subset of expr2.
 *
 * @param expr1 The first expression to check.
 * @param expr2 The second expression to check.
 * @param assums1 The assumptions for the first expression.
 * @param assums2 The assumptions for the second expression.
 * @return true if expr1 is a subset of expr2, false otherwise.
 */
bool is_subset(const MultiExpression& expr1, const MultiExpression& expr2,
               const Assumptions& assums1, const Assumptions& assums2);

/**
 * @brief Interprets the expressions as integer sets and checks if expr1 is disjoint from expr2.
 *
 * @param expr1 The first expression to check.
 * @param expr2 The second expression to check.
 * @param assums1 The assumptions for the first expression.
 * @param assums2 The assumptions for the second expression.
 * @return true if expr1 is disjoint from expr2, false otherwise.
 */
bool is_disjoint(const MultiExpression& expr1, const MultiExpression& expr2,
                 const Assumptions& assums1, const Assumptions& assums2);

}  // namespace symbolic
}  // namespace sdfg
