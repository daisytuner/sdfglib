#include "sdfg/symbolic/utils.h"

#include <isl/ctx.h>
#include <isl/set.h>
#include <isl/space.h>

#include <regex>

#include "sdfg/codegen/language_extensions/c_language_extension.h"
#include "sdfg/symbolic/assumptions.h"
#include "sdfg/symbolic/extreme_values.h"
#include "sdfg/symbolic/polynomials.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace symbolic {

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
        if (assums.find(sym) == assums.end() && assums.at(sym).constant()) {
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
    for (auto& dim : dimensions) {
        auto sym = symbolic::symbol(dim);
        auto map = assums.at(sym).map();
        if (map == SymEngine::null) {
            continue;
        }
        if (!SymEngine::is_a<SymEngine::Add>(*map)) {
            continue;
        }
        auto args = SymEngine::rcp_static_cast<const SymEngine::Add>(map)->get_args();
        if (args.size() != 2) {
            continue;
        }
        auto arg0 = args[0];
        auto arg1 = args[1];
        if (!symbolic::eq(arg0, symbolic::symbol(dim))) {
            arg0 = args[1];
            arg1 = args[0];
        }
        if (!symbolic::eq(arg0, symbolic::symbol(dim))) {
            continue;
        }
        if (!SymEngine::is_a<SymEngine::Integer>(*arg1)) {
            continue;
        }
        if (symbolic::eq(arg1, symbolic::one())) {
            continue;
        }
        auto lb = assums.at(sym).lower_bound();
        if (!SymEngine::is_a<SymEngine::Integer>(*lb)) {
            continue;
        }

        std::string iter = "__daisy_iterator_" + dim;
        std::string con = "exists " + iter + " : " + dim + " = " + language_extension.expression(lb) + " + " + iter +
                          " * " + language_extension.expression(arg1);
        constraints.push_back(con);
    }
    if (!constraints.empty()) {
        map += " : ";
        map += helpers::join(constraints, " and ");
    }

    map += " }";

    // NV Symbols, e.g., threadIdx.x
    map = std::regex_replace(map, std::regex("\\."), "_");

    // min, max
    map = std::regex_replace(map, std::regex("__daisy_min"), "min_");
    map = std::regex_replace(map, std::regex("__daisy_max"), "max_");

    return map;
}

std::tuple<std::string, std::string, std::string> expressions_to_intersection_map_str(
    const MultiExpression& expr1,
    const MultiExpression& expr2,
    const Symbol& indvar,
    const Assumptions& assums1,
    const Assumptions& assums2
) {
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
        if (sym->get_name() != indvar->get_name() && assums1.at(sym).constant() && assums2.at(sym).constant()) {
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
    auto constraints_syms_1 = generate_constraints(syms, assums1, seen);
    seen.clear();
    auto constraints_syms_2 = generate_constraints(syms, assums2, seen);

    // Extend parameters with additional symbols from constraints
    for (auto& con : constraints_syms_1) {
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
    for (auto& con : constraints_syms_2) {
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
    // Add bounds
    for (auto& con : constraints_syms_1) {
        auto con_1 = con;
        for (auto& iter : dimensions) {
            con_1 = symbolic::subs(con_1, symbolic::symbol(iter), symbolic::symbol(iter + "_1"));
        }
        auto con_str_1 = constraint_to_isl_str(con_1);
        if (con_str_1.empty()) {
            continue;
        }
        constraints_1.push_back(con_str_1);
    }
    for (auto& dim : dimensions) {
        auto sym = symbolic::symbol(dim);
        auto map = assums1.at(sym).map();
        if (map == SymEngine::null) {
            continue;
        }
        if (!SymEngine::is_a<SymEngine::Add>(*map)) {
            continue;
        }
        auto args = SymEngine::rcp_static_cast<const SymEngine::Add>(map)->get_args();
        if (args.size() != 2) {
            continue;
        }
        auto arg0 = args[0];
        auto arg1 = args[1];
        if (!symbolic::eq(arg0, symbolic::symbol(dim))) {
            arg0 = args[1];
            arg1 = args[0];
        }
        if (!symbolic::eq(arg0, symbolic::symbol(dim))) {
            continue;
        }
        if (!SymEngine::is_a<SymEngine::Integer>(*arg1)) {
            continue;
        }
        if (symbolic::eq(arg1, symbolic::one())) {
            continue;
        }
        auto lb = assums2.at(sym).lower_bound();
        if (!SymEngine::is_a<SymEngine::Integer>(*lb)) {
            continue;
        }

        std::string dim1 = dim + "_1";
        std::string iter = "__daisy_iterator_" + dim1;
        std::string con = "exists " + iter + " : " + dim1 + " = " + language_extension.expression(lb) + " + " + iter +
                          " * " + language_extension.expression(arg1);
        constraints_1.push_back(con);
    }
    if (!constraints_1.empty()) {
        map_1 += " : ";
        map_1 += helpers::join(constraints_1, " and ");
    }
    map_1 += " }";

    std::vector<std::string> constraints_2;
    for (auto& con : constraints_syms_2) {
        auto con_2 = con;
        for (auto& iter : dimensions) {
            con_2 = symbolic::subs(con_2, symbolic::symbol(iter), symbolic::symbol(iter + "_2"));
        }
        auto con_str_2 = constraint_to_isl_str(con_2);
        if (con_str_2.empty()) {
            continue;
        }
        constraints_2.push_back(con_str_2);
    }
    for (auto& dim : dimensions) {
        auto sym = symbolic::symbol(dim);
        auto map = assums2.at(sym).map();
        if (map == SymEngine::null) {
            continue;
        }
        if (!SymEngine::is_a<SymEngine::Add>(*map)) {
            continue;
        }
        auto args = SymEngine::rcp_static_cast<const SymEngine::Add>(map)->get_args();
        if (args.size() != 2) {
            continue;
        }
        auto arg0 = args[0];
        auto arg1 = args[1];
        if (!symbolic::eq(arg0, symbolic::symbol(dim))) {
            arg0 = args[1];
            arg1 = args[0];
        }
        if (!symbolic::eq(arg0, symbolic::symbol(dim))) {
            continue;
        }
        if (!SymEngine::is_a<SymEngine::Integer>(*arg1)) {
            continue;
        }
        if (symbolic::eq(arg1, symbolic::one())) {
            continue;
        }
        auto lb = assums2.at(sym).lower_bound();
        if (!SymEngine::is_a<SymEngine::Integer>(*lb)) {
            continue;
        }

        std::string dim2 = dim + "_2";
        std::string iter = "__daisy_iterator_" + dim2;
        std::string con = "exists " + iter + " : " + dim2 + " = " + language_extension.expression(lb) + " + " + iter +
                          " * " + language_extension.expression(arg1);
        constraints_2.push_back(con);
    }
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
    if (dimensions_syms.find(indvar) != dimensions_syms.end()) {
        monotonicity_constraints.push_back(indvar->get_name() + "_1 != " + indvar->get_name() + "_2");
    }
    if (!monotonicity_constraints.empty()) {
        map_3 += " : ";
        map_3 += helpers::join(monotonicity_constraints, " and ");
    }
    map_3 += " }";

    map_1 = std::regex_replace(map_1, std::regex("\\."), "_");
    map_2 = std::regex_replace(map_2, std::regex("\\."), "_");
    map_3 = std::regex_replace(map_3, std::regex("\\."), "_");

    map_1 = std::regex_replace(map_1, std::regex("__daisy_min"), "min");
    map_1 = std::regex_replace(map_1, std::regex("__daisy_max"), "max");
    map_2 = std::regex_replace(map_2, std::regex("__daisy_min"), "min");
    map_2 = std::regex_replace(map_2, std::regex("__daisy_max"), "max");
    map_3 = std::regex_replace(map_3, std::regex("__daisy_min"), "min");
    map_3 = std::regex_replace(map_3, std::regex("__daisy_max"), "max");

    return {map_1, map_2, map_3};
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
        if (SymEngine::is_a<SymEngine::Infty>(*lhs) || SymEngine::is_a<SymEngine::Infty>(*rhs)) {
            return "";
        }
        auto res = language_extension.expression(lhs) + " < " + language_extension.expression(rhs);
        res = std::regex_replace(res, std::regex("\\."), "_");
        res = std::regex_replace(res, std::regex("__daisy_min"), "min");
        res = std::regex_replace(res, std::regex("__daisy_max"), "max");
        return res;
    } else if (SymEngine::is_a<SymEngine::LessThan>(*con)) {
        auto le = SymEngine::rcp_static_cast<const SymEngine::LessThan>(con);
        auto lhs = le->get_arg1();
        auto rhs = le->get_arg2();
        if (SymEngine::is_a<SymEngine::Infty>(*lhs) || SymEngine::is_a<SymEngine::Infty>(*rhs)) {
            return "";
        }
        auto res = language_extension.expression(lhs) + " <= " + language_extension.expression(rhs);
        res = std::regex_replace(res, std::regex("\\."), "_");
        res = std::regex_replace(res, std::regex("__daisy_min"), "min");
        res = std::regex_replace(res, std::regex("__daisy_max"), "max");
        return res;
    } else if (SymEngine::is_a<SymEngine::Equality>(*con)) {
        auto eq = SymEngine::rcp_static_cast<const SymEngine::Equality>(con);
        auto lhs = eq->get_arg1();
        auto rhs = eq->get_arg2();
        if (SymEngine::is_a<SymEngine::Infty>(*lhs) || SymEngine::is_a<SymEngine::Infty>(*rhs)) {
            return "";
        }
        auto res = language_extension.expression(lhs) + " == " + language_extension.expression(rhs);
        res = std::regex_replace(res, std::regex("\\."), "_");
        res = std::regex_replace(res, std::regex("__daisy_min"), "min");
        res = std::regex_replace(res, std::regex("__daisy_max"), "max");
        return res;
    } else if (SymEngine::is_a<SymEngine::Unequality>(*con)) {
        auto ne = SymEngine::rcp_static_cast<const SymEngine::Unequality>(con);
        auto lhs = ne->get_arg1();
        auto rhs = ne->get_arg2();
        if (SymEngine::is_a<SymEngine::Infty>(*lhs) || SymEngine::is_a<SymEngine::Infty>(*rhs)) {
            return "";
        }
        auto res = language_extension.expression(lhs) + " != " + language_extension.expression(rhs);
        res = std::regex_replace(res, std::regex("\\."), "_");
        res = std::regex_replace(res, std::regex("__daisy_min"), "min");
        res = std::regex_replace(res, std::regex("__daisy_max"), "max");
        return res;
    }

    return "";
}

void canonicalize_map_dims(isl_map* map, const std::string& in_prefix, const std::string& out_prefix) {
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

MultiExpression delinearize(const MultiExpression& expr, const Assumptions& assums) {
    MultiExpression delinearized;
    for (auto& dim : expr) {
        // Step 1: Convert expression into an affine polynomial
        SymbolVec symbols;
        for (auto& sym : atoms(dim)) {
            if (assums.at(sym).constant() && assums.at(sym).map() == SymEngine::null) {
                continue;
            }
            symbols.push_back(sym);
        }
        if (symbols.empty() || symbols.size() <= 1) {
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
            remaining = symbolic::expand(remaining);
            remaining = symbolic::simplify(remaining);

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
            auto ub_stride = maximum(stride, {}, assums);
            auto ub_remaining = maximum(remaining, {}, assums);
            auto cond_stride = symbolic::Ge(ub_stride, ub_remaining);
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

} // namespace symbolic
} // namespace sdfg
