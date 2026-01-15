#include "sdfg/symbolic/series.h"

#include "sdfg/symbolic/extreme_values.h"
#include "sdfg/symbolic/polynomials.h"

namespace sdfg {
namespace symbolic {
namespace series {

std::pair<bool, std::pair<Integer, Integer>>
affine_int_coeffs(const Expression expr, const Symbol sym, const Assumptions& assums) {
    SymbolVec symbols = {sym};
    auto poly = polynomial(expr, symbols);
    if (poly == SymEngine::null) {
        return {false, {integer(0), integer(0)}};
    }
    auto coeffs = affine_coefficients(poly, symbols);
    if (coeffs.empty()) {
        return {false, {integer(0), integer(0)}};
    }
    auto mul = minimum(coeffs[sym], {}, assums);
    if (mul == SymEngine::null) {
        return {false, {integer(0), integer(0)}};
    }
    auto offset = minimum(coeffs[symbol("__daisy_constant__")], {}, assums);
    if (offset == SymEngine::null) {
        return {false, {integer(0), integer(0)}};
    }
    if (!SymEngine::is_a<SymEngine::Integer>(*mul) || !SymEngine::is_a<SymEngine::Integer>(*offset)) {
        return {false, {integer(0), integer(0)}};
    }
    auto mul_int = SymEngine::rcp_dynamic_cast<const SymEngine::Integer>(mul);
    auto offset_int = SymEngine::rcp_dynamic_cast<const SymEngine::Integer>(offset);
    return {true, {mul_int, offset_int}};
}

bool is_monotonic_affine(const Expression expr, const Symbol sym, const Assumptions& assums) {
    auto [success, coeffs] = affine_int_coeffs(expr, sym, assums);
    if (!success) {
        return false;
    }
    auto [mul_int, offset_int] = coeffs;
    try {
        signed long mul_value = mul_int->as_int();
        signed long offset_value = offset_int->as_int();
        if (mul_value > 0 && offset_value > 0) {
            return true;
        }
    } catch (SymEngine::SymEngineException&) {
        return false;
    }

    return false;
}

bool is_contiguous_affine(const Expression expr, const Symbol sym, const Assumptions& assums) {
    auto [success, coeffs] = affine_int_coeffs(expr, sym, assums);
    if (!success) {
        return false;
    }
    auto [mul_int, offset_int] = coeffs;
    try {
        signed long mul_value = mul_int->as_int();
        signed long offset_value = offset_int->as_int();
        if (mul_value == 1 && offset_value == 1) {
            return true;
        }
    } catch (SymEngine::SymEngineException&) {
        return false;
    }

    return false;
}

bool is_monotonic_pow(const Expression expr, const Symbol sym, const Assumptions& assums) {
    if (SymEngine::is_a<SymEngine::Pow>(*expr)) {
        auto pow = SymEngine::rcp_dynamic_cast<const SymEngine::Pow>(expr);
        auto base = pow->get_base();
        auto exp = pow->get_exp();
        if (SymEngine::is_a<SymEngine::Integer>(*exp) && SymEngine::is_a<SymEngine::Symbol>(*base)) {
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

bool is_monotonic(const Expression expr, const Symbol sym, const Assumptions& assums) {
    if (is_monotonic_affine(expr, sym, assums)) {
        return true;
    }
    return is_monotonic_pow(expr, sym, assums);
}

bool is_contiguous(const Expression expr, const Symbol sym, const Assumptions& assums) {
    if (is_contiguous_affine(expr, sym, assums)) {
        return true;
    }
    return false;
}

} // namespace series
} // namespace symbolic
} // namespace sdfg
