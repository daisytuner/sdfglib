#pragma once

#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace symbolic {

typedef SymEngine::RCP<const SymEngine::MExprPoly> Polynomial;
typedef std::unordered_map<Symbol, Expression, SymEngine::RCPBasicHash, SymEngine::RCPBasicKeyEq> AffineCoeffs;

/**
 * @brief Converts an Expression to a Polynomial.
 *
 * @param expr The Expression to convert.
 * @param symbols A vector of symbols that will be used in the polynomial.
 *
 * @return A Polynomial representation of the Expression.
 */
Polynomial polynomial(const Expression expr, SymbolVec& symbols);

/**
 * @brief Converts a Polynomial of degree 1 to a AffineCoeffs.
 * @param poly The Polynomial to convert.
 * @param symbols A vector of symbols that will be used in the coefficients map.
 *
 * @return A AffineCoeffs where the keys are symbols and the values are their corresponding coefficients.
 */
AffineCoeffs affine_coefficients(Polynomial poly, SymbolVec& symbols);

/// Create the inverse function for AffineCoeffs
Expression affine_inverse(AffineCoeffs coeffs, Symbol symbol);

std::pair<Expression, Expression> polynomial_div(const Expression& dividend, const Expression& divisor);


} // namespace symbolic
} // namespace sdfg
