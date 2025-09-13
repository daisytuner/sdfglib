#pragma once

#include "sdfg/symbolic/assumptions.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace symbolic {
namespace series {

bool is_monotonic(const Expression expr, const Symbol sym, const Assumptions& assums);

bool is_contiguous(const Expression expr, const Symbol sym, const Assumptions& assums);

} // namespace series
} // namespace symbolic
} // namespace sdfg
