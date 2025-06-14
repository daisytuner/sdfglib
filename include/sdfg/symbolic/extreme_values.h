#pragma once

#include "sdfg/symbolic/assumptions.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace symbolic {

/**
 * @brief Compute the minimum of an expression as max of candidates given a set of assumptions.
 *
 * @param expr The expression to compute the minimum of.
 * @param assumptions The assumptions to use.
 * @return The minimum of the expression.
 */
Expression minimum(const Expression& expr, const Assumptions& assumptions);

/**
 * @brief Compute the maximum of an expression as min of candidates given a set of assumptions.
 *
 * @param expr The expression to compute the maximum of.
 * @param assumptions The assumptions to use.
 * @return The maximum of the expression.
 */
Expression maximum(const Expression& expr, const Assumptions& assumptions);

}  // namespace symbolic
}  // namespace sdfg
