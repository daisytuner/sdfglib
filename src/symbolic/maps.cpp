#include "sdfg/symbolic/maps.h"

#include <isl/ctx.h>
#include <isl/set.h>
#include <isl/space.h>

#include "sdfg/symbolic/extreme_values.h"
#include "sdfg/symbolic/polynomials.h"
#include "sdfg/symbolic/utils.h"

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

bool is_disjoint_isl(const MultiExpression& expr1, const MultiExpression& expr2,
                     const Symbol& indvar, const Assumptions& assums1, const Assumptions& assums2) {
    if (expr1.size() != expr2.size()) {
        return false;
    }
    if (expr1.empty()) {
        return false;
    }

    isl_ctx* ctx = isl_ctx_alloc();

    // Transform both expressions into two maps with separate dimensions
    auto expr1_delinearized = delinearize(expr1, assums1);
    auto expr2_delinearized = delinearize(expr2, assums2);
    auto maps = expressions_to_intersection_map_str(expr1_delinearized, expr2_delinearized, indvar,
                                                    assums1, assums2);
    isl_map* map_1 = isl_map_read_from_str(ctx, std::get<0>(maps).c_str());
    isl_map* map_2 = isl_map_read_from_str(ctx, std::get<1>(maps).c_str());
    isl_map* map_3 = isl_map_read_from_str(ctx, std::get<2>(maps).c_str());
    if (!map_1 || !map_2 || !map_3) {
        if (map_1) {
            isl_map_free(map_1);
        }
        if (map_2) {
            isl_map_free(map_2);
        }
        if (map_3) {
            isl_map_free(map_3);
        }
        isl_ctx_free(ctx);
        return false;
    }

    // Find aliasing pairs under the constraint that dimensions are different

    isl_map* composed = isl_map_apply_domain(map_2, map_3);
    if (!composed) {
        isl_map_free(map_1);
        if (map_2) {
            isl_map_free(map_2);
        }
        if (map_3) {
            isl_map_free(map_3);
        }
        isl_ctx_free(ctx);
        return false;
    }
    isl_map* alias_pairs = isl_map_intersect(composed, map_1);
    if (!alias_pairs) {
        if (composed) {
            isl_map_free(composed);
        }
        if (map_1) {
            isl_map_free(map_1);
        }
        isl_ctx_free(ctx);
        return false;
    }

    bool disjoint = isl_map_is_empty(alias_pairs);
    isl_map_free(alias_pairs);
    isl_ctx_free(ctx);

    return disjoint;
}

bool is_disjoint_monotonic(const MultiExpression& expr1, const MultiExpression& expr2,
                           const Symbol& indvar, const Assumptions& assums1,
                           const Assumptions& assums2) {
    for (size_t i = 0; i < expr1.size(); i++) {
        auto& dim1 = expr1[i];
        if (expr2.size() <= i) {
            continue;
        }
        auto& dim2 = expr2[i];
        if (!symbolic::eq(dim1, dim2)) {
            continue;
        }

        // Collect all symbols
        symbolic::SymbolSet syms;
        for (auto& sym : symbolic::atoms(dim1)) {
            syms.insert(sym);
        }

        // Collect all non-constant symbols
        bool can_analyze = true;
        for (auto& sym : syms) {
            if (!is_parameter(sym, assums1)) {
                if (sym->get_name() != indvar->get_name()) {
                    can_analyze = false;
                    break;
                }
            }
        }
        if (!can_analyze) {
            continue;
        }

        // Check if both dimensions are monotonic in non-constant symbols
        if (symbolic::is_monotonic(dim1, indvar, assums1)) {
            return true;
        }
    }

    return false;
}

bool intersects(const MultiExpression& expr1, const MultiExpression& expr2, const Symbol& indvar,
                const Assumptions& assums1, const Assumptions& assums2) {
    if (is_disjoint_monotonic(expr1, expr2, indvar, assums1, assums2)) {
        return false;
    }
    return !is_disjoint_isl(expr1, expr2, indvar, assums1, assums2);
}

}  // namespace symbolic
}  // namespace sdfg