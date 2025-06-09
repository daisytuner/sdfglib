#include "sdfg/symbolic/functions.h"

#include "sdfg/symbolic/extreme_values.h"
#include "sdfg/symbolic/polynomials.h"

namespace sdfg {
namespace symbolic {

bool is_monotonic(const Expression& expr, const Symbol& sym, const Assumptions& assums) {
    SymbolVec symbols = {sym};
    auto poly = polynomial(expr, symbols);
    if (poly == SymEngine::null) {
        return false;
    }
    auto coeffs = affine_coefficients(poly, symbols);
    if (coeffs.empty()) {
        return false;
    }
    auto mul = minimum(coeffs[sym], assums);
    if (mul == SymEngine::null) {
        return false;
    }
    auto offset = minimum(coeffs[symbol("__daisy_constant__")], assums);
    if (offset == SymEngine::null) {
        return false;
    }
    if (!SymEngine::is_a<SymEngine::Integer>(*mul) ||
        !SymEngine::is_a<SymEngine::Integer>(*offset)) {
        return false;
    }
    auto mul_int = SymEngine::rcp_dynamic_cast<const SymEngine::Integer>(mul);
    auto offset_int = SymEngine::rcp_dynamic_cast<const SymEngine::Integer>(offset);
    if (mul_int->as_int() <= 0 || offset_int->as_int() <= 0) {
        return false;
    }

    return true;
}

bool is_contiguous(const Expression& expr, const Symbol& sym, const Assumptions& assums) {
    SymbolVec symbols = {sym};
    auto poly = polynomial(expr, symbols);
    if (poly == SymEngine::null) {
        return false;
    }
    auto coeffs = affine_coefficients(poly, symbols);
    if (coeffs.empty()) {
        return false;
    }
    auto mul = minimum(coeffs[sym], assums);
    if (mul == SymEngine::null) {
        return false;
    }
    auto offset = minimum(coeffs[symbol("__daisy_constant__")], assums);
    if (offset == SymEngine::null) {
        return false;
    }
    if (!SymEngine::is_a<SymEngine::Integer>(*mul) ||
        !SymEngine::is_a<SymEngine::Integer>(*offset)) {
        return false;
    }
    auto mul_int = SymEngine::rcp_dynamic_cast<const SymEngine::Integer>(mul);
    auto offset_int = SymEngine::rcp_dynamic_cast<const SymEngine::Integer>(offset);
    if (mul_int->as_int() == 1 && offset_int->as_int() == 1) {
        return false;
    }

    return true;
}

}  // namespace symbolic
}  // namespace sdfg