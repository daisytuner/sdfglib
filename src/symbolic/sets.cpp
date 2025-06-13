#include "sdfg/symbolic/sets.h"

#include <isl/ctx.h>
#include <isl/map.h>
#include <isl/set.h>
#include <isl/space.h>

#include <regex>

#include "sdfg/codegen/language_extensions/c_language_extension.h"

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
    map_3 += "] : ";
    for (size_t i = 0; i < dimensions.size(); i++) {
        map_3 += dimensions[i] + "_1 != " + dimensions[i] + "_2";
        if (i < dimensions.size() - 1) {
            map_3 += " and ";
        }
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

bool is_disjoint(const MultiExpression& expr1, const MultiExpression& expr2,
                 const SymbolSet& params, const Assumptions& assums) {
    if (expr1.size() != expr2.size()) {
        return false;
    }

    isl_ctx* ctx = isl_ctx_alloc();

    // Transform both expressions into two maps with separate dimensions
    auto maps = expressions_to_intersection_map_str(expr1, expr2, params, assums);
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

    // (consumes map_1)
    isl_map* rev_map1 = isl_map_reverse(map_1);
    if (!rev_map1) {
        if (map_1) {
            isl_map_free(map_1);
        }
        isl_map_free(map_2);
        isl_map_free(map_3);
        isl_ctx_free(ctx);
        return false;
    }

    // (consumes rev_map1 and map_2)
    isl_map* alias_pairs = isl_map_apply_range(rev_map1, map_2);
    if (!alias_pairs) {
        if (rev_map1) {
            isl_map_free(rev_map1);
        }
        if (map_2) {
            isl_map_free(map_2);
        }
        isl_map_free(map_3);
        isl_ctx_free(ctx);
        return false;
    }
    isl_map* alias_pairs_2 = isl_map_intersect(alias_pairs, map_3);
    if (!alias_pairs_2) {
        if (alias_pairs) {
            isl_map_free(alias_pairs);
        }
        if (map_3) {
            isl_map_free(map_3);
        }
        isl_ctx_free(ctx);
        return false;
    }

    bool disjoint = isl_map_is_empty(alias_pairs_2);
    isl_map_free(alias_pairs_2);
    isl_ctx_free(ctx);

    return disjoint;
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
    std::string map_str = expressions_to_diagonal_map_str(expr1, expr2, params, assums);

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

}  // namespace symbolic
}  // namespace sdfg