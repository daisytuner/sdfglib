#pragma once

#include <string>
#include <vector>

#include "sdfg/symbolic/assumptions.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace symbolic {

typedef std::vector<Expression> MultiExpression;

bool intersect(const MultiExpression& expr1, const SymbolSet& params1, const MultiExpression& expr2,
               const SymbolSet& params2, const Assumptions& assums);

bool is_equivalent(const MultiExpression& expr1, const MultiExpression& expr2,
                   const SymbolSet& params, const Assumptions& assums);

std::string expression_to_isl_map_str(const MultiExpression& expr, const SymbolSet& params,
                                      const Assumptions& assums);

std::string expressions_to_isl_map_str(const MultiExpression& expr1, const MultiExpression& expr2,
                                       const SymbolSet& params, const Assumptions& assums);

ExpressionSet generate_constraints(SymbolSet& syms, const Assumptions& assums, SymbolSet& seen);

std::string constraint_to_isl_str(const Expression& con);

}  // namespace symbolic
}  // namespace sdfg
