#pragma once

#include <string>
#include <vector>

#include "sdfg/symbolic/assumptions.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace symbolic {

typedef std::vector<Expression> MultiExpression;

bool is_disjoint(const MultiExpression& expr1, const MultiExpression& expr2,
                 const SymbolSet& params, const Assumptions& assums);

bool is_equivalent(const MultiExpression& expr1, const MultiExpression& expr2,
                   const SymbolSet& params, const Assumptions& assums);

MultiExpression delinearize(const MultiExpression& expr, const SymbolSet& params,
                            const Assumptions& assums);

std::tuple<std::string, std::string, std::string> expressions_to_intersection_map_str(
    const MultiExpression& expr1, const MultiExpression& expr2, const SymbolSet& params,
    const Assumptions& assums);

std::string expressions_to_diagonal_map_str(const MultiExpression& expr1,
                                            const MultiExpression& expr2, const SymbolSet& params,
                                            const Assumptions& assums);

ExpressionSet generate_constraints(SymbolSet& syms, const Assumptions& assums, SymbolSet& seen);

std::string constraint_to_isl_str(const Expression& con);

}  // namespace symbolic
}  // namespace sdfg
