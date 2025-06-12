#pragma once

#include "sdfg/symbolic/assumptions.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace symbolic {

typedef std::vector<Expression> MultiExpression;

bool intersect(const MultiExpression& expr1, const SymbolSet& params1, const MultiExpression& expr2,
               const SymbolSet& params2, const Assumptions& assums);

}  // namespace symbolic
}  // namespace sdfg
