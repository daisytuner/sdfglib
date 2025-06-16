#pragma once

#include <unordered_map>

#include "sdfg/symbolic/assumptions.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace symbolic {

/**
 * @brief Check if an expression is monotonic w.r.t. a symbol.
 *
 * @param expr The expression to check.
 * @param sym The symbol to check.
 * @param parameters A set of symbols to treat as parameters (not simplified).
 * @param assums A set of assumptions about bounds of symbols.
 * @return True if the expression is monotonic w.r.t. the symbol, false otherwise.
 */
bool is_monotonic(const Expression& expr, const Symbol& sym, const SymbolSet& parameters,
                  const Assumptions& assums);

/**
 * @brief Check if an expression is contiguous w.r.t. a symbol.
 *
 * @param expr The expression to check.
 * @param sym The symbol to check.
 * @param parameters A set of symbols to treat as parameters (not simplified).
 * @param assums A set of assumptions about bounds of symbols.
 * @return True if the expression is contiguous w.r.t. the symbol, false otherwise.
 */
bool is_contiguous(const Expression& expr, const Symbol& sym, const SymbolSet& parameters,
                   const Assumptions& assums);

}  // namespace symbolic
}  // namespace sdfg
