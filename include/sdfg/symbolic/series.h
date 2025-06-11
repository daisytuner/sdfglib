#pragma once

#include <unordered_map>

#include "sdfg/symbolic/assumptions.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace symbolic {

bool is_monotonic(const Expression& expr, const Symbol& sym, const Assumptions& assums);

bool is_contiguous(const Expression& expr, const Symbol& sym, const Assumptions& assums);

}  // namespace symbolic
}  // namespace sdfg
