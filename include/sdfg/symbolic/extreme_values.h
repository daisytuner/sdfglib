#pragma once

#include "sdfg/symbolic/assumptions.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace symbolic {

/**
 * @brief Compute the minimum of an expression.
 *
 * @param expr The expression to compute the minimum of.
 * @param parameters A set of symbols to treat as parameters.
 * @param assumptions A set of assumptions about bounds of symbols.
 * @return The minimum of the expression, or null if the expression is not bounded.
 */
Expression minimum(const Expression expr, const SymbolSet& parameters, const Assumptions& assumptions);

/**
 * @brief Compute the maximum of an expression.
 *
 * @param expr The expression to compute the maximum of.
 * @param parameters A set of symbols to treat as parameters.
 * @param assumptions A set of assumptions about bounds of symbols.
 * @return The maximum of the expression, or null if the expression is not bounded.
 */
Expression maximum(const Expression expr, const SymbolSet& parameters, const Assumptions& assumptions);

/**
 * @brief Compute the minimum of an expression.
 *
 * @param expr The expression to compute the minimum of.
 * @param parameters A set of symbols to treat as parameters.
 * @param assumptions A set of assumptions about bounds of symbols.
 * @return The minimum of the expression, or null if the expression is not bounded.
 */
Expression minimum_new(const Expression expr, const SymbolSet& parameters, const Assumptions& assumptions, bool tight);

/**
 * @brief Compute the maximum of an expression.
 *
 * @param expr The expression to compute the maximum of.
 * @param parameters A set of symbols to treat as parameters.
 * @param assumptions A set of assumptions about bounds of symbols.
 * @return The maximum of the expression, or null if the expression is not bounded.
 */
Expression maximum_new(const Expression expr, const SymbolSet& parameters, const Assumptions& assumptions, bool tight);

} // namespace symbolic
} // namespace sdfg
