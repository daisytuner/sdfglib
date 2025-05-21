#pragma once

#include "sdfg/symbolic/assumptions.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace symbolic {

enum Sign { POSITIVE, NEGATIVE, NONE };

typedef SymEngine::RCP<const SymEngine::UExprPoly> Polynomial;
typedef SymEngine::RCP<const SymEngine::MExprPoly> MultiPolynomial;
typedef std::pair<Expression, Expression> Affine;
typedef std::unordered_map<Symbol, int, SymEngine::RCPBasicHash, SymEngine::RCPBasicKeyEq> AffineCoefficients;

Polynomial polynomial(const Expression& expr, const Symbol& symbol);

MultiPolynomial multi_polynomial(const Expression& expr, SymbolicVector& symbols);

AffineCoefficients affine_coefficients(MultiPolynomial& poly, SymbolicVector& symbols);

Affine affine(const Expression& expr, const Symbol& symbol);

Sign strict_monotonicity_affine(const Expression& func, const Symbol& symbol);

Sign strict_monotonicity_affine(const Expression& func, const symbolic::Assumptions& assumptions);

Sign strict_monotonicity_affine(const Expression& func, const Symbol& symbol,
                                const symbolic::Assumptions& assumptions);

Sign strict_monotonicity(const Expression& func, const Symbol& symbol);

Sign strict_monotonicity(const Expression& func, const symbolic::Assumptions& assumptions);

Sign strict_monotonicity(const Expression& func, const Symbol& symbol,
                         const symbolic::Assumptions& assumptions);

Sign strict_monotonicity(const Expression& func);

bool contiguity_affine(const Expression& func, const Symbol& symbol);

bool contiguity_affine(const Expression& func, const Symbol& symbol,
                       const symbolic::Assumptions& assumptions);

bool contiguity(const Expression& func, const Symbol& symbol);

bool contiguity(const Expression& func, const Symbol& symbol,
                const symbolic::Assumptions& assumptions);

Expression inverse(const Symbol& lhs, const Expression& rhs);

bool contains_infinity(const Expression& expr);

Expression lower_bound_analysis(const Expression& expr, symbolic::Assumptions& assumptions);

Expression upper_bound_analysis(const Expression& expr, symbolic::Assumptions& assumptions);
}  // namespace symbolic
}  // namespace sdfg
