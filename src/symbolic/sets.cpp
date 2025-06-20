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

MultiExpression delinearize(const MultiExpression& expr, const Assumptions& assums);

bool is_monotonic(const Symbol& sym, const Assumptions& assums);

bool is_parameter(const Symbol& sym, const Assumptions& assums);

std::string expression_to_map_str(const MultiExpression& expr, const Assumptions& assums);

ExpressionSet generate_constraints(SymbolSet& syms, const Assumptions& assums, SymbolSet& seen);

std::string constraint_to_isl_str(const Expression& con);

void canonicalize_map_dims(isl_map* map, const std::string& in_prefix,
                           const std::string& out_prefix) {
    int n_in = isl_map_dim(map, isl_dim_in);
    int n_out = isl_map_dim(map, isl_dim_out);

    for (int i = 0; i < n_in; ++i) {
        std::string name = in_prefix + std::to_string(i);
        map = isl_map_set_dim_name(map, isl_dim_in, i, name.c_str());
    }

    for (int i = 0; i < n_out; ++i) {
        std::string name = out_prefix + std::to_string(i);
        map = isl_map_set_dim_name(map, isl_dim_out, i, name.c_str());
    }
}

bool is_subset(const MultiExpression& expr1, const MultiExpression& expr2,
               const Assumptions& assums1, const Assumptions& assums2) {
    if (expr1.size() == 0 && expr2.size() == 0) {
        return true;
    }

    auto expr1_delinearized = delinearize(expr1, assums1);
    auto expr2_delinearized = delinearize(expr2, assums2);

    std::string map_1_str = expression_to_map_str(expr1_delinearized, assums1);
    std::string map_2_str = expression_to_map_str(expr2_delinearized, assums2);

    isl_ctx* ctx = isl_ctx_alloc();
    isl_map* map_1 = isl_map_read_from_str(ctx, map_1_str.c_str());
    isl_map* map_2 = isl_map_read_from_str(ctx, map_2_str.c_str());
    if (!map_1 || !map_2) {
        if (map_1) {
            isl_map_free(map_1);
        }
        if (map_2) {
            isl_map_free(map_2);
        }
        isl_ctx_free(ctx);
        return false;
    }
    isl_space* params_map1 = isl_space_params(isl_map_get_space(map_1));
    isl_space* params_map2 = isl_space_params(isl_map_get_space(map_2));

    // Align parameters carefully:
    isl_space* unified_params =
        isl_space_align_params(isl_space_copy(params_map1), isl_space_copy(params_map2));

    // Align maps to unified params:
    isl_map* aligned_map_1 = isl_map_align_params(map_1, isl_space_copy(unified_params));
    isl_map* aligned_map_2 = isl_map_align_params(map_2, isl_space_copy(unified_params));

    // Remove parameters explicitly (project them out)
    aligned_map_1 = isl_map_project_out(aligned_map_1, isl_dim_param, 0,
                                        isl_map_dim(aligned_map_1, isl_dim_param));
    aligned_map_2 = isl_map_project_out(aligned_map_2, isl_dim_param, 0,
                                        isl_map_dim(aligned_map_2, isl_dim_param));

    canonicalize_map_dims(aligned_map_1, "in_", "out_");
    canonicalize_map_dims(aligned_map_2, "in_", "out_");

    isl_set* set_1 = isl_map_range(aligned_map_1);
    isl_set* set_2 = isl_map_range(aligned_map_2);

    bool subset = isl_set_is_subset(set_1, set_2) == isl_bool_true;

    // Cleanup carefully:
    isl_map_free(aligned_map_1);
    isl_map_free(aligned_map_2);
    isl_space_free(unified_params);
    isl_space_free(params_map1);
    isl_space_free(params_map2);
    isl_ctx_free(ctx);

    return subset;
}

bool is_disjoint(const MultiExpression& expr1, const MultiExpression& expr2,
                 const Assumptions& assums1, const Assumptions& assums2) {
    auto expr1_delinearized = delinearize(expr1, assums1);
    auto expr2_delinearized = delinearize(expr2, assums2);

    std::string map_1_str = expression_to_map_str(expr1_delinearized, assums1);
    std::string map_2_str = expression_to_map_str(expr2_delinearized, assums2);

    isl_ctx* ctx = isl_ctx_alloc();
    isl_map* map_1 = isl_map_read_from_str(ctx, map_1_str.c_str());
    isl_map* map_2 = isl_map_read_from_str(ctx, map_2_str.c_str());
    if (!map_1 || !map_2) {
        if (map_1) isl_map_free(map_1);
        if (map_2) isl_map_free(map_2);
        isl_ctx_free(ctx);
        return false;
    }

    isl_space* params_map1 = isl_space_params(isl_map_get_space(map_1));
    isl_space* params_map2 = isl_space_params(isl_map_get_space(map_2));

    // Align parameters carefully:
    isl_space* unified_params =
        isl_space_align_params(isl_space_copy(params_map1), isl_space_copy(params_map2));

    // Align maps to unified params:
    isl_map* aligned_map_1 = isl_map_align_params(map_1, isl_space_copy(unified_params));
    isl_map* aligned_map_2 = isl_map_align_params(map_2, isl_space_copy(unified_params));

    // Remove parameters explicitly (project them out)
    aligned_map_1 = isl_map_project_out(aligned_map_1, isl_dim_param, 0,
                                        isl_map_dim(aligned_map_1, isl_dim_param));
    aligned_map_2 = isl_map_project_out(aligned_map_2, isl_dim_param, 0,
                                        isl_map_dim(aligned_map_2, isl_dim_param));

    canonicalize_map_dims(aligned_map_1, "in_", "out_");
    canonicalize_map_dims(aligned_map_2, "in_", "out_");

    isl_set* set_1 = isl_map_range(aligned_map_1);
    isl_set* set_2 = isl_map_range(aligned_map_2);

    bool disjoint = isl_set_is_disjoint(set_1, set_2) == isl_bool_true;

    // Cleanup carefully:
    isl_map_free(aligned_map_1);
    isl_map_free(aligned_map_2);
    isl_space_free(unified_params);
    isl_space_free(params_map1);
    isl_space_free(params_map2);
    isl_ctx_free(ctx);

    return disjoint;
}

MultiExpression delinearize(const MultiExpression& expr, const Assumptions& assums) {
    MultiExpression delinearized;
    for (auto& dim : expr) {
        // Step 1: Convert expression into an affine polynomial
        SymbolVec symbols;
        for (auto& sym : atoms(dim)) {
            if (!is_parameter(sym, assums)) {
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

bool is_parameter(const Symbol& sym, const Assumptions& assums) {
    if (assums.find(sym) == assums.end()) {
        return false;
    }
    auto& ass = assums.at(sym);
    if (ass.map() == SymEngine::null) {
        return false;
    }
    if (symbolic::eq(ass.map(), symbolic::zero())) {
        return true;
    }
    return false;
}

bool is_monotonic(const Symbol& sym, const Assumptions& assums) {
    if (assums.find(sym) == assums.end()) {
        return false;
    }
    auto& ass = assums.at(sym);
    if (ass.map() == SymEngine::null) {
        return false;
    }
    if (symbolic::eq(ass.map(), symbolic::add(sym, symbolic::one()))) {
        return true;
    }
    return false;
}

std::string expression_to_map_str(const MultiExpression& expr, const Assumptions& assums) {
    codegen::CLanguageExtension language_extension;

    // Get all symbols
    symbolic::SymbolSet syms;
    for (auto& expr : expr) {
        auto syms_expr = symbolic::atoms(expr);
        syms.insert(syms_expr.begin(), syms_expr.end());
    }

    // Distinguish between dimensions and parameters
    std::vector<std::string> dimensions;
    SymbolSet dimensions_syms;
    std::vector<std::string> parameters;
    SymbolSet parameters_syms;
    for (auto& sym : syms) {
        if (is_parameter(sym, assums)) {
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
        std::sort(parameters.begin(), parameters.end());
        map += "[";
        map += helpers::join(parameters, ", ");
        map += "] -> ";
    }
    map += "{ [" + helpers::join(dimensions, ", ") + "] -> [";
    for (size_t i = 0; i < expr.size(); i++) {
        auto dim = expr[i];
        map += language_extension.expression(dim);
        if (i < expr.size() - 1) {
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

}  // namespace symbolic
}  // namespace sdfg