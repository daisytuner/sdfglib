#include "sdfg/symbolic/extreme_values.h"

#include "sdfg/symbolic/polynomials.h"

namespace sdfg {
namespace symbolic {

size_t MAX_DEPTH = 100;

Expression minimum(const Expression& expr, const Assumptions& assumptions, const size_t depth) {
    // Base Cases
    if (depth > MAX_DEPTH) {
        return expr;
    }

    if (SymEngine::is_a<SymEngine::Integer>(*expr)) {
        return expr;
    } else if (SymEngine::is_a<SymEngine::Infty>(*expr)) {
        return expr;
    } else if (SymEngine::is_a<SymEngine::NaN>(*expr)) {
        return expr;
    } else if (SymEngine::is_a<SymEngine::Max>(*expr)) {
        auto args = SymEngine::rcp_dynamic_cast<const SymEngine::Max>(expr)->get_args();
        Expression lbs = SymEngine::null;
        for (const auto& arg : args) {
            auto lb = minimum(arg, assumptions, depth + 1);
            if (lb == SymEngine::null) {
                return SymEngine::null;
            }
            if (lbs == SymEngine::null) {
                lbs = lb;
            } else {
                lbs = symbolic::min(lbs, lb);
            }
        }
        return lbs;
    } else if (SymEngine::is_a<SymEngine::Min>(*expr)) {
        auto args = SymEngine::rcp_dynamic_cast<const SymEngine::Min>(expr)->get_args();
        Expression lbs = SymEngine::null;
        for (const auto& arg : args) {
            auto lb = minimum(arg, assumptions, depth + 1);
            if (lb == SymEngine::null) {
                return SymEngine::null;
            }
            if (lbs == SymEngine::null) {
                lbs = lb;
            } else {
                lbs = symbolic::min(lbs, lb);
            }
        }
        return lbs;
    }

    // Symbol
    if (SymEngine::is_a<SymEngine::Symbol>(*expr)) {
        auto sym = SymEngine::rcp_static_cast<const SymEngine::Symbol>(expr);
        if (assumptions.find(sym) != assumptions.end()) {
            return minimum(assumptions.at(sym).lower_bound(), assumptions, depth + 1);
        }
        return sym;
    }

    // Expression
    symbolic::SymbolVec symbols;
    for (const auto& symbol : atoms(expr)) {
        symbols.push_back(symbol);
    }
    auto poly = polynomial(expr, symbols);
    if (poly == SymEngine::null) {
        return SymEngine::null;
    }

    auto coeffs = affine_coefficients(poly, symbols);
    if (coeffs.empty()) {
        return SymEngine::null;
    }

    Expression lb = symbolic::zero();
    for (const auto& symbol : symbols) {
        auto lb_coeff = minimum(coeffs[symbol], assumptions, depth + 1);
        if (lb_coeff == SymEngine::null) {
            return SymEngine::null;
        }
        auto lb_symbol = minimum(symbol, assumptions, depth + 1);
        if (lb_symbol == SymEngine::null) {
            return SymEngine::null;
        }
        lb = symbolic::add(lb, symbolic::mul(lb_coeff, lb_symbol));
    }
    auto lb_const = minimum(coeffs[symbolic::symbol("__daisy_constant__")], assumptions, depth + 1);
    if (lb_const == SymEngine::null) {
        return SymEngine::null;
    }
    lb = symbolic::add(lb, lb_const);
    return lb;
}

Expression maximum(const Expression& expr, const Assumptions& assumptions, const size_t depth) {
    if (depth > MAX_DEPTH) {
        return expr;
    }

    // Base Cases
    if (SymEngine::is_a<SymEngine::Integer>(*expr)) {
        return expr;
    } else if (SymEngine::is_a<SymEngine::Infty>(*expr)) {
        return expr;
    } else if (SymEngine::is_a<SymEngine::NaN>(*expr)) {
        return expr;
    } else if (SymEngine::is_a<SymEngine::Max>(*expr)) {
        auto args = SymEngine::rcp_dynamic_cast<const SymEngine::Max>(expr)->get_args();
        Expression ubs = SymEngine::null;
        for (const auto& arg : args) {
            auto ub = maximum(arg, assumptions, depth + 1);
            if (ub == SymEngine::null) {
                return SymEngine::null;
            }
            if (ubs == SymEngine::null) {
                ubs = ub;
            } else {
                ubs = symbolic::max(ubs, ub);
            }
        }
        return ubs;
    } else if (SymEngine::is_a<SymEngine::Min>(*expr)) {
        auto args = SymEngine::rcp_dynamic_cast<const SymEngine::Min>(expr)->get_args();
        Expression ubs = SymEngine::null;
        for (const auto& arg : args) {
            auto ub = maximum(arg, assumptions, depth + 1);
            if (ub == SymEngine::null) {
                return SymEngine::null;
            }
            if (ubs == SymEngine::null) {
                ubs = ub;
            } else {
                ubs = symbolic::max(ubs, ub);
            }
        }
        return ubs;
    }

    // Symbol
    if (SymEngine::is_a<SymEngine::Symbol>(*expr)) {
        auto sym = SymEngine::rcp_static_cast<const SymEngine::Symbol>(expr);
        if (assumptions.find(sym) != assumptions.end()) {
            return maximum(assumptions.at(sym).upper_bound(), assumptions, depth + 1);
        }
        return sym;
    }

    // Expression
    symbolic::SymbolVec symbols;
    for (const auto& symbol : atoms(expr)) {
        symbols.push_back(symbol);
    }
    auto poly = polynomial(expr, symbols);
    if (poly == SymEngine::null) {
        return SymEngine::null;
    }

    auto coeffs = affine_coefficients(poly, symbols);
    if (coeffs.empty()) {
        return SymEngine::null;
    }

    Expression ub = symbolic::zero();
    for (const auto& symbol : symbols) {
        auto ub_coeff = maximum(coeffs[symbol], assumptions, depth + 1);
        if (ub_coeff == SymEngine::null) {
            return SymEngine::null;
        }
        auto ub_symbol = maximum(symbol, assumptions, depth + 1);
        if (ub_symbol == SymEngine::null) {
            return SymEngine::null;
        }
        ub = symbolic::add(ub, symbolic::mul(ub_coeff, ub_symbol));
    }
    auto ub_const = maximum(coeffs[symbolic::symbol("__daisy_constant__")], assumptions, depth + 1);
    if (ub_const == SymEngine::null) {
        return SymEngine::null;
    }
    ub = symbolic::add(ub, ub_const);
    return ub;
}

Expression minimum(const Expression& expr, const Assumptions& assumptions) {
    auto lb = minimum(expr, assumptions, 0);
    if (lb == SymEngine::null) {
        return symbolic::infty(-1);
    }
    return lb;
}

Expression maximum(const Expression& expr, const Assumptions& assumptions) {
    auto ub = maximum(expr, assumptions, 0);
    if (ub == SymEngine::null) {
        return symbolic::infty(1);
    }
    return ub;
}

}  // namespace symbolic
}  // namespace sdfg
