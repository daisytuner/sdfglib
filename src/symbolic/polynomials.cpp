#include "sdfg/symbolic/polynomials.h"

#include <symengine/polys/basic_conversions.h>

namespace sdfg {
namespace symbolic {

Polynomial polynomial(const Expression expr, SymbolVec& symbols) {
    try {
        ExpressionSet gens;
        for (auto& symbol : symbols) {
            gens.insert(symbol);
        }
        return SymEngine::from_basic<SymEngine::MExprPoly>(expr, gens);
    } catch (SymEngine::SymEngineException& e) {
        return SymEngine::null;
    }
};

AffineCoeffs affine_coefficients(Polynomial poly, SymbolVec& symbols) {
    AffineCoeffs coeffs;
    for (auto& symbol : symbols) {
        coeffs[symbol] = symbolic::zero();
    }
    coeffs[symbolic::symbol("__daisy_constant__")] = symbolic::zero();

    auto& D = poly->get_poly().get_dict();
    for (auto& [exponents, coeff] : D) {
        // Check if sum of exponents is <= 1
        symbolic::Symbol symbol = symbolic::symbol("__daisy_constant__");
        unsigned total_deg = 0;
        for (size_t i = 0; i < exponents.size(); i++) {
            auto& e = exponents[i];
            if (e > 0) {
                symbol = symbols[i];
            }
            total_deg += e;
        }
        if (total_deg > 1) {
            return {};
        }

        // Add coefficient to corresponding symbol
        coeffs[symbol] = symbolic::add(coeffs[symbol], coeff);
    }

    return coeffs;
}

} // namespace symbolic
} // namespace sdfg
