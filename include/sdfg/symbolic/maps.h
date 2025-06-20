#pragma once

#include <unordered_map>

#include "sdfg/symbolic/assumptions.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace symbolic {

bool intersects(const MultiExpression& expr1, const MultiExpression& expr2, const Symbol& indvar,
                const Assumptions& assums1, const Assumptions& assums2);

bool is_monotonic(const Expression& expr, const Symbol& sym, const Assumptions& assums);

bool is_contiguous(const Expression& expr, const Symbol& sym, const Assumptions& assums);

}  // namespace symbolic
}  // namespace sdfg
