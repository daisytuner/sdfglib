#include "sdfg/symbolic/extreme_values.h"

#include "sdfg/symbolic/polynomials.h"

namespace sdfg {
namespace symbolic {

size_t MAX_DEPTH = 100;
Expression minimum(const Expression& expr, const SymbolSet& parameters,
                   const Assumptions& assumptions, const size_t depth);
Expression maximum(const Expression& expr, const SymbolSet& parameters,
                   const Assumptions& assumptions, const size_t depth);

Expression minimum(const Expression& expr, const SymbolSet& parameters,
                   const Assumptions& assumptions, const size_t depth) {
    // Base Cases
    if (depth > MAX_DEPTH) {
        return expr;
    }

    if (SymEngine::is_a<SymEngine::NaN>(*expr)) {
        return SymEngine::null;
    }

    if (SymEngine::is_a<SymEngine::Integer>(*expr)) {
        return expr;
    } else if (SymEngine::is_a<SymEngine::Infty>(*expr)) {
        return expr;
    }

    // Symbol
    if (SymEngine::is_a<SymEngine::Symbol>(*expr)) {
        auto sym = SymEngine::rcp_static_cast<const SymEngine::Symbol>(expr);
        if (parameters.find(sym) != parameters.end()) {
            return sym;
        }
        if (assumptions.find(sym) != assumptions.end()) {
            return minimum(assumptions.at(sym).lower_bound(), parameters, assumptions, depth + 1);
        }
        return SymEngine::null;
    }

    // Mul
    if (SymEngine::is_a<SymEngine::Mul>(*expr)) {
        auto mul = SymEngine::rcp_static_cast<const SymEngine::Mul>(expr);
        const auto& args = mul->get_args();
        if (args.size() != 2) {
            // Extend to N-ary multiplication if needed
            return SymEngine::null;
        }

        Expression a = args[0];
        Expression b = args[1];

        Expression a_min = minimum(a, parameters, assumptions, depth + 1);
        Expression a_max = maximum(a, parameters, assumptions, depth + 1);
        Expression b_min = minimum(b, parameters, assumptions, depth + 1);
        Expression b_max = maximum(b, parameters, assumptions, depth + 1);

        if (a_min == SymEngine::null || a_max == SymEngine::null || b_min == SymEngine::null ||
            b_max == SymEngine::null) {
            return SymEngine::null;
        }

        // Compute all 4 combinations
        Expression p1 = symbolic::mul(a_min, b_min);
        Expression p2 = symbolic::mul(a_min, b_max);
        Expression p3 = symbolic::mul(a_max, b_min);
        Expression p4 = symbolic::mul(a_max, b_max);

        // Return minimum of all products
        return symbolic::min(symbolic::min(p1, p2), symbolic::min(p3, p4));
    }

    // Add
    if (SymEngine::is_a<SymEngine::Add>(*expr)) {
        auto add = SymEngine::rcp_static_cast<const SymEngine::Add>(expr);
        const auto& args = add->get_args();
        Expression lbs = SymEngine::null;
        for (const auto& arg : args) {
            auto lb = minimum(arg, parameters, assumptions, depth + 1);
            if (lb == SymEngine::null) {
                return SymEngine::null;
            }
            if (lbs == SymEngine::null) {
                lbs = lb;
            } else {
                lbs = symbolic::add(lbs, lb);
            }
        }
        return lbs;
    }

    // Max
    if (SymEngine::is_a<SymEngine::Max>(*expr)) {
        auto args = SymEngine::rcp_dynamic_cast<const SymEngine::Max>(expr)->get_args();
        Expression lbs = SymEngine::null;
        for (const auto& arg : args) {
            auto lb = minimum(arg, parameters, assumptions, depth + 1);
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

    // Min
    if (SymEngine::is_a<SymEngine::Min>(*expr)) {
        auto args = SymEngine::rcp_dynamic_cast<const SymEngine::Min>(expr)->get_args();
        Expression lbs = SymEngine::null;
        for (const auto& arg : args) {
            auto lb = minimum(arg, parameters, assumptions, depth + 1);
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

    return SymEngine::null;
}

Expression maximum(const Expression& expr, const SymbolSet& parameters,
                   const Assumptions& assumptions, const size_t depth) {
    if (depth > MAX_DEPTH) {
        return expr;
    }

    if (SymEngine::is_a<SymEngine::NaN>(*expr)) {
        return SymEngine::null;
    }

    // Base Cases
    if (SymEngine::is_a<SymEngine::Integer>(*expr)) {
        return expr;
    } else if (SymEngine::is_a<SymEngine::Infty>(*expr)) {
        return expr;
    }

    // Symbol
    if (SymEngine::is_a<SymEngine::Symbol>(*expr)) {
        auto sym = SymEngine::rcp_static_cast<const SymEngine::Symbol>(expr);
        if (parameters.find(sym) != parameters.end()) {
            return sym;
        }
        if (assumptions.find(sym) != assumptions.end()) {
            return maximum(assumptions.at(sym).upper_bound(), parameters, assumptions, depth + 1);
        }
        return SymEngine::null;
    }

    // Mul
    if (SymEngine::is_a<SymEngine::Mul>(*expr)) {
        auto mul = SymEngine::rcp_static_cast<const SymEngine::Mul>(expr);
        const auto& args = mul->get_args();
        if (args.size() != 2) {
            // Extend to N-ary multiplication if needed
            return SymEngine::null;
        }

        Expression a = args[0];
        Expression b = args[1];

        Expression a_min = minimum(a, parameters, assumptions, depth + 1);
        Expression a_max = maximum(a, parameters, assumptions, depth + 1);
        Expression b_min = minimum(b, parameters, assumptions, depth + 1);
        Expression b_max = maximum(b, parameters, assumptions, depth + 1);

        if (a_min == SymEngine::null || a_max == SymEngine::null || b_min == SymEngine::null ||
            b_max == SymEngine::null) {
            return SymEngine::null;
        }

        // Compute all 4 combinations
        Expression p1 = symbolic::mul(a_min, b_min);
        Expression p2 = symbolic::mul(a_min, b_max);
        Expression p3 = symbolic::mul(a_max, b_min);
        Expression p4 = symbolic::mul(a_max, b_max);

        // Return maximum of all products
        return symbolic::max(symbolic::max(p1, p2), symbolic::max(p3, p4));
    }

    // Add
    if (SymEngine::is_a<SymEngine::Add>(*expr)) {
        auto add = SymEngine::rcp_static_cast<const SymEngine::Add>(expr);
        const auto& args = add->get_args();
        Expression ubs = SymEngine::null;
        for (const auto& arg : args) {
            auto ub = maximum(arg, parameters, assumptions, depth + 1);
            if (ub == SymEngine::null) {
                return SymEngine::null;
            }
            if (ubs == SymEngine::null) {
                ubs = ub;
            } else {
                ubs = symbolic::add(ubs, ub);
            }
        }
        return ubs;
    }

    // Max
    if (SymEngine::is_a<SymEngine::Max>(*expr)) {
        auto args = SymEngine::rcp_dynamic_cast<const SymEngine::Max>(expr)->get_args();
        Expression ubs = SymEngine::null;
        for (const auto& arg : args) {
            auto ub = maximum(arg, parameters, assumptions, depth + 1);
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

    // Min
    if (SymEngine::is_a<SymEngine::Min>(*expr)) {
        auto args = SymEngine::rcp_dynamic_cast<const SymEngine::Min>(expr)->get_args();
        Expression ubs = SymEngine::null;
        for (const auto& arg : args) {
            auto ub = maximum(arg, parameters, assumptions, depth + 1);
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

    return SymEngine::null;
}

Expression minimum(const Expression& expr, const SymbolSet& parameters,
                   const Assumptions& assumptions) {
    return minimum(expr, parameters, assumptions, 0);
}

Expression maximum(const Expression& expr, const SymbolSet& parameters,
                   const Assumptions& assumptions) {
    return maximum(expr, parameters, assumptions, 0);
}

}  // namespace symbolic
}  // namespace sdfg
