#include "sdfg/symbolic/sets.h"

#include <isl/ctx.h>
#include <isl/map.h>
#include <isl/set.h>
#include <isl/space.h>

#include <regex>

#include "sdfg/codegen/language_extensions/c_language_extension.h"
#include "sdfg/symbolic/extreme_values.h"
#include "sdfg/symbolic/polynomials.h"
#include "sdfg/symbolic/series.h"

namespace sdfg {
namespace symbolic {

ExpressionSet generate_constraints(SymbolSet& syms, const Assumptions& assums, SymbolSet& seen) {
    ExpressionSet constraints;
    for (auto& sym : syms) {
        if (assums.find(sym) == assums.end()) {
            continue;
        }
        if (seen.find(sym) != seen.end()) {
            continue;
        }
        seen.insert(sym);

        auto ub = assums.at(sym).upper_bound();
        auto lb = assums.at(sym).lower_bound();
        if (!symbolic::eq(ub, symbolic::infty(1))) {
            if (SymEngine::is_a<SymEngine::Min>(*ub)) {
                auto min = SymEngine::rcp_static_cast<const SymEngine::Min>(ub);
                auto args = min->get_args();
                for (auto& arg : args) {
                    auto con = symbolic::Le(sym, arg);
                    auto con_syms = symbolic::atoms(con);
                    constraints.insert(con);

                    auto con_cons = generate_constraints(con_syms, assums, seen);
                    constraints.insert(con_cons.begin(), con_cons.end());
                }
            } else {
                auto con = symbolic::Le(sym, ub);
                auto con_syms = symbolic::atoms(con);
                constraints.insert(con);

                auto con_cons = generate_constraints(con_syms, assums, seen);
                constraints.insert(con_cons.begin(), con_cons.end());
            }
        }
        if (!symbolic::eq(lb, symbolic::infty(-1))) {
            if (SymEngine::is_a<SymEngine::Max>(*lb)) {
                auto max = SymEngine::rcp_static_cast<const SymEngine::Max>(lb);
                auto args = max->get_args();
                for (auto& arg : args) {
                    auto con = symbolic::Ge(sym, arg);
                    auto con_syms = symbolic::atoms(con);
                    constraints.insert(con);

                    auto con_cons = generate_constraints(con_syms, assums, seen);
                    constraints.insert(con_cons.begin(), con_cons.end());
                }
            } else {
                auto con = symbolic::Ge(sym, lb);
                auto con_syms = symbolic::atoms(con);
                constraints.insert(con);

                auto con_cons = generate_constraints(con_syms, assums, seen);
                constraints.insert(con_cons.begin(), con_cons.end());
            }
        }
    }
    return constraints;
}

std::string constraint_to_isl_str(const Expression& con) {
    codegen::CLanguageExtension language_extension;

    if (SymEngine::is_a<SymEngine::StrictLessThan>(*con)) {
        auto le = SymEngine::rcp_static_cast<const SymEngine::StrictLessThan>(con);
        auto lhs = le->get_arg1();
        auto rhs = le->get_arg2();
        return language_extension.expression(lhs) + " < " + language_extension.expression(rhs);
    } else if (SymEngine::is_a<SymEngine::LessThan>(*con)) {
        auto le = SymEngine::rcp_static_cast<const SymEngine::LessThan>(con);
        auto lhs = le->get_arg1();
        auto rhs = le->get_arg2();
        return language_extension.expression(lhs) + " <= " + language_extension.expression(rhs);
    } else if (SymEngine::is_a<SymEngine::Equality>(*con)) {
        auto eq = SymEngine::rcp_static_cast<const SymEngine::Equality>(con);
        auto lhs = eq->get_arg1();
        auto rhs = eq->get_arg2();
        return language_extension.expression(lhs) + " == " + language_extension.expression(rhs);
    } else if (SymEngine::is_a<SymEngine::Unequality>(*con)) {
        auto ne = SymEngine::rcp_static_cast<const SymEngine::Unequality>(con);
        auto lhs = ne->get_arg1();
        auto rhs = ne->get_arg2();
        return language_extension.expression(lhs) + " != " + language_extension.expression(rhs);
    }

    return "";
}

std::tuple<std::string, std::string, std::string> expressions_to_intersection_map_str(
    const MultiExpression& expr1, const MultiExpression& expr2, const SymbolSet& params,
    const SymbolSet& monotonics, const Assumptions& assums) {
    codegen::CLanguageExtension language_extension;

    // Get all symbols
    symbolic::SymbolSet syms;
    for (auto& expr : expr1) {
        auto syms_expr = symbolic::atoms(expr);
        syms.insert(syms_expr.begin(), syms_expr.end());
    }
    for (auto& expr : expr2) {
        auto syms_expr = symbolic::atoms(expr);
        syms.insert(syms_expr.begin(), syms_expr.end());
    }

    // Distinguish between dimensions and parameters
    std::vector<std::string> dimensions;
    SymbolSet dimensions_syms;
    std::vector<std::string> parameters;
    SymbolSet parameters_syms;
    for (auto& sym : syms) {
        if (params.find(sym) != params.end()) {
            if (parameters_syms.find(sym) != parameters_syms.end()) {
                continue;
            }
            parameters.push_back(sym->get_name());
            parameters_syms.insert(sym);
        } else {
            if (dimensions_syms.find(sym) != dimensions_syms.end()) {
                continue;
            }
            dimensions.push_back(sym->get_name());
            dimensions_syms.insert(sym);
        }
    }

    // Generate constraints
    SymbolSet seen;
    auto constraints_syms = generate_constraints(syms, assums, seen);

    // Extend parameters with additional symbols from constraints
    for (auto& con : constraints_syms) {
        auto con_syms = symbolic::atoms(con);
        for (auto& con_sym : con_syms) {
            if (dimensions_syms.find(con_sym) == dimensions_syms.end()) {
                if (parameters_syms.find(con_sym) != parameters_syms.end()) {
                    continue;
                }
                parameters.push_back(con_sym->get_name());
                parameters_syms.insert(con_sym);
            }
        }
    }

    // Define two maps
    std::string map_1;
    std::string map_2;
    if (!parameters.empty()) {
        map_1 += "[";
        map_1 += helpers::join(parameters, ", ");
        map_1 += "] -> ";
        map_2 += "[";
        map_2 += helpers::join(parameters, ", ");
        map_2 += "] -> ";
    }
    map_1 += "{ [";
    map_2 += "{ [";
    for (size_t i = 0; i < dimensions.size(); i++) {
        map_1 += dimensions[i] + "_1";
        map_2 += dimensions[i] + "_2";
        if (i < dimensions.size() - 1) {
            map_1 += ", ";
            map_2 += ", ";
        }
    }
    map_1 += "] -> [";
    map_2 += "] -> [";
    for (size_t i = 0; i < expr1.size(); i++) {
        auto dim = expr1[i];
        for (auto& iter : dimensions) {
            dim = symbolic::subs(dim, symbolic::symbol(iter), symbolic::symbol(iter + "_1"));
        }
        map_1 += language_extension.expression(dim);
        if (i < expr1.size() - 1) {
            map_1 += ", ";
        }
    }
    for (size_t i = 0; i < expr2.size(); i++) {
        auto dim = expr2[i];
        for (auto& iter : dimensions) {
            dim = symbolic::subs(dim, symbolic::symbol(iter), symbolic::symbol(iter + "_2"));
        }
        map_2 += language_extension.expression(dim);
        if (i < expr2.size() - 1) {
            map_2 += ", ";
        }
    }
    map_1 += "] ";
    map_2 += "] ";

    std::vector<std::string> constraints_1;
    std::vector<std::string> constraints_2;
    // Add bounds
    for (auto& con : constraints_syms) {
        auto con_1 = con;
        auto con_2 = con;
        for (auto& iter : dimensions) {
            con_1 = symbolic::subs(con_1, symbolic::symbol(iter), symbolic::symbol(iter + "_1"));
            con_2 = symbolic::subs(con_2, symbolic::symbol(iter), symbolic::symbol(iter + "_2"));
        }
        auto con_str_1 = constraint_to_isl_str(con_1);
        if (con_str_1.empty()) {
            continue;
        }
        auto con_str_2 = constraint_to_isl_str(con_2);
        if (!con_str_1.empty()) {
            constraints_1.push_back(con_str_1);
            constraints_2.push_back(con_str_2);
        }
    }
    if (!constraints_1.empty()) {
        map_1 += " : ";
        map_1 += helpers::join(constraints_1, " and ");
    }
    map_1 += " }";

    if (!constraints_2.empty()) {
        map_2 += " : ";
        map_2 += helpers::join(constraints_2, " and ");
    }
    map_2 += " }";

    std::string map_3 = "{ [";
    for (size_t i = 0; i < dimensions.size(); i++) {
        map_3 += dimensions[i] + "_2";
        if (i < dimensions.size() - 1) {
            map_3 += ", ";
        }
    }
    map_3 += "] -> [";
    for (size_t i = 0; i < dimensions.size(); i++) {
        map_3 += dimensions[i] + "_1";
        if (i < dimensions.size() - 1) {
            map_3 += ", ";
        }
    }
    map_3 += "]";
    std::vector<std::string> monotonicity_constraints;
    // Monotonicity constraints
    for (size_t i = 0; i < dimensions.size(); i++) {
        if (monotonics.find(symbolic::symbol(dimensions[i])) == monotonics.end()) {
            continue;
        }
        monotonicity_constraints.push_back(dimensions[i] + "_1 != " + dimensions[i] + "_2");
    }
    if (!monotonics.empty()) {
        map_3 += " : ";
        map_3 += helpers::join(monotonicity_constraints, " and ");
    }
    map_3 += " }";

    map_1 = std::regex_replace(map_1, std::regex("\\."), "_");
    map_2 = std::regex_replace(map_2, std::regex("\\."), "_");
    map_3 = std::regex_replace(map_3, std::regex("\\."), "_");

    return {map_1, map_2, map_3};
}

std::string expressions_to_diagonal_map_str(const MultiExpression& expr1,
                                            const MultiExpression& expr2, const SymbolSet& params,
                                            const Assumptions& assums) {
    codegen::CLanguageExtension language_extension;

    // Get all symbols
    symbolic::SymbolSet syms;
    for (auto& expr : expr1) {
        auto syms_expr = symbolic::atoms(expr);
        syms.insert(syms_expr.begin(), syms_expr.end());
    }
    for (auto& expr : expr2) {
        auto syms_expr = symbolic::atoms(expr);
        syms.insert(syms_expr.begin(), syms_expr.end());
    }

    // Distinguish between dimensions and parameters
    std::vector<std::string> dimensions;
    SymbolSet dimensions_syms;
    std::vector<std::string> parameters;
    SymbolSet parameters_syms;
    for (auto& sym : syms) {
        if (params.find(sym) != params.end()) {
            if (parameters_syms.find(sym) != parameters_syms.end()) {
                continue;
            }
            parameters.push_back(sym->get_name());
            parameters_syms.insert(sym);
        } else {
            if (dimensions_syms.find(sym) != dimensions_syms.end()) {
                continue;
            }
            dimensions.push_back(sym->get_name());
            dimensions_syms.insert(sym);
        }
    }

    // Generate constraints
    SymbolSet seen;
    auto constraints_syms = generate_constraints(syms, assums, seen);

    // Extend parameters with additional symbols from constraints
    for (auto& con : constraints_syms) {
        auto con_syms = symbolic::atoms(con);
        for (auto& con_sym : con_syms) {
            if (dimensions_syms.find(con_sym) == dimensions_syms.end()) {
                if (parameters_syms.find(con_sym) != parameters_syms.end()) {
                    continue;
                }
                parameters.push_back(con_sym->get_name());
                parameters_syms.insert(con_sym);
            }
        }
    }

    // Define map
    std::string map;
    if (!parameters.empty()) {
        map += "[";
        map += helpers::join(parameters, ", ");
        map += "] -> ";
    }
    map += "{ [" + helpers::join(dimensions, ", ") + "] -> [";
    for (size_t i = 0; i < expr1.size(); i++) {
        auto dim = expr1[i];
        map += language_extension.expression(dim);
        map += ", ";
    }
    for (size_t i = 0; i < expr2.size(); i++) {
        auto dim = expr2[i];
        map += language_extension.expression(dim);
        if (i < expr2.size() - 1) {
            map += ", ";
        }
    }
    map += "] ";

    std::vector<std::string> constraints;
    for (auto& con : constraints_syms) {
        auto con_str = constraint_to_isl_str(con);
        if (!con_str.empty()) {
            constraints.push_back(con_str);
        }
    }
    if (!constraints.empty()) {
        map += " : ";
        map += helpers::join(constraints, " and ");
    }

    map += " }";

    map = std::regex_replace(map, std::regex("\\."), "_");
    return map;
}

bool is_disjoint_isl(const MultiExpression& expr1, const MultiExpression& expr2,
                     const SymbolSet& params, const SymbolSet& monotonics,
                     const Assumptions& assums) {
    if (expr1.size() != expr2.size()) {
        return false;
    }
    if (expr1.empty()) {
        return false;
    }

    isl_ctx* ctx = isl_ctx_alloc();

    // Transform both expressions into two maps with separate dimensions
    auto expr1_delinearized = delinearize(expr1, params, assums);
    auto expr2_delinearized = delinearize(expr2, params, assums);
    auto maps = expressions_to_intersection_map_str(expr1, expr2, params, monotonics, assums);
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
                           const SymbolSet& params, const SymbolSet& monotonics,
                           const Assumptions& assums) {
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
        symbolic::SymbolSet generators;
        for (auto& sym : syms) {
            if (params.find(sym) == params.end()) {
                generators.insert(sym);
            }
        }
        if (generators.empty()) {
            continue;
        }

        // Check if all non-constant symbols are monotonics
        bool can_analyze = true;
        for (auto& sym : generators) {
            if (monotonics.find(sym) == monotonics.end()) {
                can_analyze = false;
                break;
            }
        }
        if (!can_analyze) {
            continue;
        }

        // Check if both dimensions are monotonic in non-constant symbols
        bool monotonic_dimension = true;
        for (auto& sym : generators) {
            if (!symbolic::is_monotonic(dim1, sym, assums)) {
                monotonic_dimension = false;
                break;
            }
        }
        if (!monotonic_dimension) {
            continue;
        }

        return true;
    }

    return false;
}

bool is_disjoint(const MultiExpression& expr1, const MultiExpression& expr2,
                 const SymbolSet& params, const SymbolSet& monotonics, const Assumptions& assums) {
    if (is_disjoint_monotonic(expr1, expr2, params, monotonics, assums)) {
        return true;
    }
    return is_disjoint_isl(expr1, expr2, params, monotonics, assums);
}

bool is_equivalent(const MultiExpression& expr1, const MultiExpression& expr2,
                   const SymbolSet& params, const Assumptions& assums) {
    if (expr1.size() != expr2.size()) {
        return false;
    }

    // Simple check for equality
    bool equals = true;
    for (size_t i = 0; i < expr1.size(); i++) {
        if (!symbolic::eq(expr1[i], expr2[i])) {
            equals = false;
            break;
        }
    }
    if (equals) {
        return true;
    }

    // Check for set equivalence
    isl_ctx* ctx = isl_ctx_alloc();
    // Builds { [params] -> [expr1..., expr2...] : assumptions }
    auto expr1_delinearized = delinearize(expr1, params, assums);
    auto expr2_delinearized = delinearize(expr2, params, assums);
    std::string map_str =
        expressions_to_diagonal_map_str(expr1_delinearized, expr2_delinearized, params, assums);
    isl_map* pair_map = isl_map_read_from_str(ctx, map_str.c_str());
    if (!pair_map) {
        isl_ctx_free(ctx);
        return false;
    }

    isl_set* range = isl_map_range(isl_map_copy(pair_map));

    // Build diagonal set: { [x₀, ..., xₙ, y₀, ..., yₙ] : x₀ = y₀ and ... and xₙ = yₙ }
    std::string diag_str = "{ [";
    for (size_t i = 0; i < expr1.size() * 2; ++i) {
        if (i > 0) diag_str += ", ";
        diag_str += "v" + std::to_string(i);
    }
    diag_str += "] : ";
    for (size_t i = 0; i < expr1.size(); ++i) {
        if (i > 0) diag_str += " and ";
        diag_str += "v" + std::to_string(i) + " = v" + std::to_string(i + expr1.size());
    }
    diag_str += " }";

    isl_set* diagonal = isl_set_read_from_str(ctx, diag_str.c_str());

    bool equivalent = false;
    if (range && diagonal) {
        equivalent = isl_set_is_subset(range, diagonal) == 1;
    }

    isl_set_free(range);
    isl_set_free(diagonal);
    isl_map_free(pair_map);
    isl_ctx_free(ctx);

    return equivalent;
}

MultiExpression delinearize(const MultiExpression& expr, const SymbolSet& params,
                            const Assumptions& assums) {
    MultiExpression delinearized;
    for (auto& dim : expr) {
        // Step 1: Convert expression into an affine polynomial
        SymbolVec symbols;
        for (auto& sym : atoms(dim)) {
            if (params.find(sym) == params.end()) {
                symbols.push_back(sym);
            }
        }
        if (symbols.empty()) {
            delinearized.push_back(dim);
            continue;
        }

        auto poly = polynomial(dim, symbols);
        if (poly == SymEngine::null) {
            delinearized.push_back(dim);
            continue;
        }

        auto aff_coeffs = affine_coefficients(poly, symbols);
        if (aff_coeffs.empty()) {
            delinearized.push_back(dim);
            continue;
        }

        // Step 2: Peel-off dimensions
        bool success = true;
        Expression remaining = dim;
        std::vector<Expression> peeled_dims;
        while (aff_coeffs.size() > 1) {
            // Find the symbol with largest stride (= largest atom count)
            Symbol new_dim = symbolic::symbol("");
            size_t max_atom_count = 0;
            for (const auto& [sym, coeff] : aff_coeffs) {
                if (sym->get_name() == "__daisy_constant__") {
                    continue;
                }
                size_t atom_count = symbolic::atoms(coeff).size();
                if (atom_count > max_atom_count || new_dim->get_name() == "") {
                    max_atom_count = atom_count;
                    new_dim = sym;
                }
            }
            if (new_dim->get_name() == "") {
                break;
            }

            // Symbol must be nonnegative
            auto sym_lb = minimum(new_dim, {}, assums);
            if (sym_lb == SymEngine::null) {
                success = false;
                break;
            }
            auto sym_cond = symbolic::Ge(sym_lb, symbolic::zero());
            if (!symbolic::is_true(sym_cond)) {
                success = false;
                break;
            }

            // Stride must be positive
            Expression stride = aff_coeffs.at(new_dim);
            auto stride_lb = minimum(stride, {}, assums);
            if (stride_lb == SymEngine::null) {
                success = false;
                break;
            }
            auto stride_cond = symbolic::Ge(stride_lb, symbolic::one());
            if (!symbolic::is_true(stride_cond)) {
                success = false;
                break;
            }

            // Peel off the dimension
            remaining = symbolic::sub(remaining, symbolic::mul(stride, new_dim));

            // Check if remainder is within bounds

            // remaining must be nonnegative
            auto rem_lb = minimum(remaining, {}, assums);
            if (rem_lb == SymEngine::null) {
                success = false;
                break;
            }
            auto cond_zero = symbolic::Ge(rem_lb, symbolic::zero());
            if (!symbolic::is_true(cond_zero)) {
                success = false;
                break;
            }

            // remaining must be less than stride
            auto rem = symbolic::sub(stride, remaining);
            rem = minimum(rem, {}, assums);
            if (rem == SymEngine::null) {
                success = false;
                break;
            }

            auto cond_stride = symbolic::Ge(rem, symbolic::one());
            if (!symbolic::is_true(cond_stride)) {
                success = false;
                break;
            }

            // Add the peeled dimension to the list
            peeled_dims.push_back(new_dim);
            aff_coeffs.erase(new_dim);
        }
        if (!success) {
            delinearized.push_back(dim);
            continue;
        }

        for (auto& peeled_dim : peeled_dims) {
            delinearized.push_back(peeled_dim);
        }

        // If remaining is not zero, then add the constant term
        if (!symbolic::eq(remaining, symbolic::zero()) && success) {
            delinearized.push_back(remaining);
        }
    }

    return delinearized;
}

bool is_subset(const MultiExpression& expr1, const MultiExpression& expr2,
               const Assumptions& assums) {
    if (expr1.size() == 0 && expr2.size() == 0) {
        return true;
    }

    SymbolSet params;
    SymbolSet monotonics;
    for (auto& entry : assums) {
        auto& ass = entry.second;

        // No knowledge about the symbol's changes
        if (ass.map() == SymEngine::null) {
            continue;
        }

        // The symbol is constant
        if (symbolic::eq(ass.map(), symbolic::zero())) {
            params.insert(entry.first);
        }

        // The symbol is monotonic
        if (symbolic::is_monotonic(ass.map(), entry.first, assums)) {
            monotonics.insert(entry.first);
        }
    }
    return is_equivalent(expr1, expr2, params, assums);
}

bool is_disjoint(const MultiExpression& expr1, const MultiExpression& expr2,
                 const Assumptions& assums) {
    if (expr1.size() == 0 && expr2.size() == 0) {
        return false;
    }

    SymbolSet params;
    SymbolSet monotonics;
    for (auto& entry : assums) {
        auto& ass = entry.second;

        // No knowledge about the symbol's changes
        if (ass.map() == SymEngine::null) {
            continue;
        }

        // The symbol is constant
        if (symbolic::eq(ass.map(), symbolic::zero())) {
            params.insert(entry.first);
        }

        // The symbol is monotonic
        if (symbolic::is_monotonic(ass.map(), entry.first, assums)) {
            monotonics.insert(entry.first);
        }
    }

    return is_disjoint(expr1, expr2, params, monotonics, assums);
}

}  // namespace symbolic
}  // namespace sdfg