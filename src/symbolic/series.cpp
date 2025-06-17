#include "sdfg/symbolic/series.h"

#include "sdfg/symbolic/extreme_values.h"
#include "sdfg/symbolic/polynomials.h"

namespace sdfg {
namespace symbolic {

bool is_monotonic_affine(const Expression& expr, const Symbol& sym, const Assumptions& assums) {
    SymbolVec symbols = {sym};
    auto poly = polynomial(expr, symbols);
    if (poly == SymEngine::null) {
        return false;
    }
    auto coeffs = affine_coefficients(poly, symbols);
    if (coeffs.empty()) {
        return false;
    }
    auto mul = minimum(coeffs[sym], {}, assums);
    if (mul == SymEngine::null) {
        return false;
    }
    auto offset = minimum(coeffs[symbol("__daisy_constant__")], {}, assums);
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

bool is_monotonic_pow(const Expression& expr, const Symbol& sym, const Assumptions& assums) {
    if (SymEngine::is_a<SymEngine::Pow>(*expr)) {
        auto pow = SymEngine::rcp_dynamic_cast<const SymEngine::Pow>(expr);
        auto base = pow->get_base();
        auto exp = pow->get_exp();
        if (SymEngine::is_a<SymEngine::Integer>(*exp) &&
            SymEngine::is_a<SymEngine::Symbol>(*base)) {
            auto exp_int = SymEngine::rcp_dynamic_cast<const SymEngine::Integer>(exp);
            if (exp_int->as_int() <= 0) {
                return false;
            }
            auto base_sym = SymEngine::rcp_static_cast<const SymEngine::Symbol>(base);
            auto ub_sym = minimum(base_sym, {}, assums);
            if (ub_sym == SymEngine::null) {
                return false;
            }
            auto positive = symbolic::Ge(ub_sym, symbolic::integer(0));
            return symbolic::is_true(positive);
        }
    }

    return false;
}

bool is_monotonic(const Expression& expr, const Symbol& sym, const Assumptions& assums) {
    if (is_monotonic_affine(expr, sym, assums)) {
        return true;
    }
    return is_monotonic_pow(expr, sym, assums);
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
    auto mul = minimum(coeffs[sym], {}, assums);
    if (mul == SymEngine::null) {
        return false;
    }
    auto offset = minimum(coeffs[symbol("__daisy_constant__")], {}, assums);
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
        return true;
    }

    return false;
}

}  // namespace symbolic
}  // namespace sdfg