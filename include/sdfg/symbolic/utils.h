#pragma once

#include <isl/map.h>

#include "sdfg/symbolic/assumptions.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace symbolic {

std::string expression_to_map_str(const MultiExpression& expr, const Assumptions& assums);

ExpressionSet generate_constraints(SymbolSet& syms, const Assumptions& assums, SymbolSet& seen);

std::string constraint_to_isl_str(const Expression& con);

void canonicalize_map_dims(isl_map* map, const std::string& in_prefix,
                           const std::string& out_prefix);

MultiExpression delinearize(const MultiExpression& expr, const Assumptions& assums);

}  // namespace symbolic
}  // namespace sdfg