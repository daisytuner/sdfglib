#include "sdfg/symbolic/sets.h"

#include <isl/ctx.h>
#include <isl/map.h>
#include <isl/set.h>
#include <isl/space.h>

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

std::string expression_to_isl_map_str(const MultiExpression& expr, const SymbolSet& params,
                                      const Assumptions& assums) {
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
        if (params.find(sym) != params.end()) {
            parameters.push_back(sym->get_name());
            parameters_syms.insert(sym);
        } else {
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

    return map;
}

std::string expressions_to_isl_map_str(const MultiExpression& expr1, const MultiExpression& expr2,
                                       const SymbolSet& params, const Assumptions& assums) {
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

    return map;
}

bool intersect(const MultiExpression& expr1, const SymbolSet& params1, const MultiExpression& expr2,
               const SymbolSet& params2, const Assumptions& assums) {
    return false;
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
    std::string map_str = expressions_to_isl_map_str(expr1, expr2, params, assums);

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