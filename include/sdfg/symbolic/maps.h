#pragma once

#include <unordered_map>

#include "sdfg/symbolic/assumptions.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace symbolic {
namespace maps {

bool is_monotonic(const Expression expr, const Symbol sym, const Assumptions& assums);

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
