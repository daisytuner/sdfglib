#include "sdfg/symbolic/extreme_values.h"

#include "sdfg/symbolic/polynomials.h"

namespace sdfg {
namespace symbolic {

size_t MAX_DEPTH = 100;
Expression minimum(const Expression expr, const SymbolSet& parameters, const Assumptions& assumptions, const size_t depth);
Expression maximum(const Expression expr, const SymbolSet& parameters, const Assumptions& assumptions, const size_t depth);

Expression minimum(const Expression expr, const SymbolSet& parameters, const Assumptions& assumptions, const size_t depth) {
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
        size_t n = args.size();

        std::vector<std::pair<Expression, Expression>> bounds;
        bounds.reserve(n);

        for (const auto& arg : args) {
            Expression min_val = minimum(arg, parameters, assumptions, depth + 1);
            Expression max_val = maximum(arg, parameters, assumptions, depth + 1);

            if (min_val == SymEngine::null || max_val == SymEngine::null) {
                return SymEngine::null;
            }
            bounds.emplace_back(min_val, max_val);
        }

        // Iterate over 2^n combinations
        Expression min_product = SymEngine::null;
        const size_t total_combinations = 1ULL << n;

        for (size_t mask = 0; mask < total_combinations; ++mask) {
            Expression product = SymEngine::integer(1);
            for (size_t i = 0; i < n; ++i) {
                const auto& bound = bounds[i];
                Expression val = (mask & (1ULL << i)) ? bound.second : bound.first;
                product = symbolic::mul(product, val);
            }
            if (min_product == SymEngine::null) {
                min_product = product;
            } else {
                min_product = symbolic::min(min_product, product);
            }
        }

        return min_product;
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

Expression maximum(const Expression expr, const SymbolSet& parameters, const Assumptions& assumptions, const size_t depth) {
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
        size_t n = args.size();

        std::vector<std::pair<Expression, Expression>> bounds;
        bounds.reserve(n);

        for (const auto& arg : args) {
            Expression min_val = minimum(arg, parameters, assumptions, depth + 1);
            Expression max_val = maximum(arg, parameters, assumptions, depth + 1);

            if (min_val == SymEngine::null || max_val == SymEngine::null) {
                return SymEngine::null;
            }
            bounds.emplace_back(min_val, max_val);
        }

        // Iterate over 2^n combinations
        Expression max_product = SymEngine::null;
        const size_t total_combinations = 1ULL << n;

        for (size_t mask = 0; mask < total_combinations; ++mask) {
            Expression product = SymEngine::integer(1);
            for (size_t i = 0; i < n; ++i) {
                const auto& bound = bounds[i];
                Expression val = (mask & (1ULL << i)) ? bound.second : bound.first;
                product = symbolic::mul(product, val);
            }
            if (max_product == SymEngine::null) {
                max_product = product;
            } else {
                max_product = symbolic::max(max_product, product);
            }
        }

        return max_product;
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

Expression minimum(const Expression expr, const SymbolSet& parameters, const Assumptions& assumptions) {
    return minimum(expr, parameters, assumptions, 0);
}

Expression maximum(const Expression expr, const SymbolSet& parameters, const Assumptions& assumptions) {
    return maximum(expr, parameters, assumptions, 0);
}

} // namespace symbolic
} // namespace sdfg
