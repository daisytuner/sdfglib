#include "sdfg/analysis/data_parallelism_analysis.h"

#include <regex>

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/codegen/language_extensions/cpp_language_extension.h"
#include "sdfg/structured_control_flow/structured_loop.h"

namespace sdfg {
namespace analysis {

std::pair<data_flow::Subset, data_flow::Subset> DataParallelismAnalysis::substitution(
    const data_flow::Subset& subset1, const data_flow::Subset& subset2, const std::string& indvar,
    const std::unordered_set<std::string>& moving_symbols, symbolic::ExpressionMap& replacements,
    std::vector<std::string>& substitions) {
    data_flow::Subset subset1_new;
    for (auto& dim : subset1) {
        auto args = dim->get_args();
        auto new_dim = dim;
        for (auto& arg : args) {
            if (!SymEngine::is_a<SymEngine::Symbol>(*arg) &&
                !SymEngine::is_a<SymEngine::Integer>(*arg)) {
                bool is_moving = false;
                for (auto& atom : symbolic::atoms(arg)) {
                    auto sym = SymEngine::rcp_static_cast<const SymEngine::Symbol>(atom);
                    if (moving_symbols.find(sym->get_name()) != moving_symbols.end()) {
                        is_moving = true;
                        break;
                    }
                }
                if (!is_moving) {
                    if (replacements.find(arg) != replacements.end()) {
                        new_dim = symbolic::subs(new_dim, arg, replacements.at(arg));
                    } else {
                        auto repl = symbolic::symbol("c_" + std::to_string(replacements.size()));
                        substitions.push_back(repl->get_name());
                        replacements.insert({arg, repl});
                        new_dim = symbolic::subs(new_dim, arg, repl);
                    }
                }
            }
        }
        subset1_new.push_back(new_dim);
    }

    data_flow::Subset subset2_new;
    for (auto& dim : subset2) {
        auto args = dim->get_args();
        auto new_dim = dim;
        for (auto& arg : args) {
            if (!SymEngine::is_a<SymEngine::Symbol>(*arg) &&
                !SymEngine::is_a<SymEngine::Integer>(*arg)) {
                bool is_moving = false;
                for (auto& atom : symbolic::atoms(arg)) {
                    auto sym = SymEngine::rcp_static_cast<const SymEngine::Symbol>(atom);
                    if (moving_symbols.find(sym->get_name()) != moving_symbols.end()) {
                        is_moving = true;
                        break;
                    }
                }
                if (!is_moving) {
                    if (replacements.find(arg) != replacements.end()) {
                        new_dim = symbolic::subs(new_dim, arg, replacements.at(arg));
                    } else {
                        auto repl = symbolic::symbol("c_" + std::to_string(replacements.size()));
                        substitions.push_back(repl->get_name());
                        replacements.insert({arg, repl});
                        new_dim = symbolic::subs(new_dim, arg, repl);
                    }
                }
            }
        }
        subset2_new.push_back(new_dim);
    }

    return {subset1_new, subset2_new};
};

std::pair<data_flow::Subset, data_flow::Subset> DataParallelismAnalysis::delinearization(
    const data_flow::Subset& subset1, const data_flow::Subset& subset2,
    const std::unordered_set<std::string>& moving_symbols,
    const symbolic::Assumptions& assumptions) {
    // Attempt to prove:
    // dim = i + j * M
    // We can delinearize iff:
    // 1. M is a constant
    // 2. i and j are not symbols
    // 3. M >= ub(i)
    data_flow::Subset subset1_new;
    for (auto& dim : subset1) {
        if (!SymEngine::is_a<SymEngine::Add>(*dim)) {
            subset1_new.push_back(dim);
            continue;
        }
        auto add = SymEngine::rcp_static_cast<const SymEngine::Add>(dim);
        if (add->get_args().size() != 2) {
            subset1_new.push_back(dim);
            continue;
        }
        auto offset = add->get_args()[0];
        auto mult = add->get_args()[1];
        if (!SymEngine::is_a<SymEngine::Mul>(*mult)) {
            auto tmp = offset;
            offset = mult;
            mult = tmp;
        }
        if (!SymEngine::is_a<SymEngine::Mul>(*mult)) {
            subset1_new.push_back(dim);
            continue;
        }

        // Offset must be a symbol and moving
        if (!SymEngine::is_a<SymEngine::Symbol>(*offset)) {
            subset1_new.push_back(dim);
            continue;
        }
        auto off = SymEngine::rcp_static_cast<const SymEngine::Symbol>(offset);
        if (moving_symbols.find(off->get_name()) == moving_symbols.end()) {
            subset1_new.push_back(dim);
            continue;
        }

        // Multiplier must be two symbols and one moving
        auto mult_ = SymEngine::rcp_static_cast<const SymEngine::Mul>(mult);
        if (mult_->get_args().size() != 2) {
            subset1_new.push_back(dim);
            continue;
        }
        auto multiplier = mult_->get_args()[0];
        auto indvar_ = mult_->get_args()[1];
        if (!SymEngine::is_a<SymEngine::Symbol>(*multiplier) ||
            !SymEngine::is_a<SymEngine::Symbol>(*indvar_)) {
            subset1_new.push_back(dim);
            continue;
        }
        auto mul = SymEngine::rcp_static_cast<const SymEngine::Symbol>(multiplier);
        auto indvar = SymEngine::rcp_static_cast<const SymEngine::Symbol>(indvar_);
        if (moving_symbols.find(mul->get_name()) != moving_symbols.end()) {
            auto tmp = mul;
            mul = indvar;
            indvar = tmp;
        }
        if (moving_symbols.find(mul->get_name()) != moving_symbols.end() ||
            moving_symbols.find(indvar->get_name()) == moving_symbols.end()) {
            subset1_new.push_back(dim);
            continue;
        }

        bool is_nonnegative = false;
        symbolic::ExpressionSet lbs_off;
        symbolic::lower_bounds(off, assumptions, lbs_off);
        for (auto& lb : lbs_off) {
            if (SymEngine::is_a<SymEngine::Integer>(*lb)) {
                auto lb_int = SymEngine::rcp_static_cast<const SymEngine::Integer>(lb);
                if (lb_int->as_int() >= 0) {
                    is_nonnegative = true;
                    break;
                }
            }
        }
        if (!is_nonnegative) {
            subset1_new.push_back(dim);
            continue;
        }

        bool success = false;
        symbolic::ExpressionSet ubs_off;
        symbolic::upper_bounds(off, assumptions, ubs_off);
        for (auto& ub_off : ubs_off) {
            if (symbolic::eq(mul, symbolic::add(ub_off, symbolic::one()))) {
                subset1_new.push_back(indvar);
                subset1_new.push_back(off);

                success = true;
                break;
            }
        }
        if (success) {
            continue;
        }
        subset1_new.push_back(dim);
    }

    data_flow::Subset subset2_new;
    for (auto& dim : subset2) {
        if (!SymEngine::is_a<SymEngine::Add>(*dim)) {
            subset2_new.push_back(dim);
            continue;
        }
        auto add = SymEngine::rcp_static_cast<const SymEngine::Add>(dim);
        if (add->get_args().size() != 2) {
            subset2_new.push_back(dim);
            continue;
        }
        auto offset = add->get_args()[0];
        auto mult = add->get_args()[1];
        if (!SymEngine::is_a<SymEngine::Mul>(*mult)) {
            auto tmp = offset;
            offset = mult;
            mult = tmp;
        }
        if (!SymEngine::is_a<SymEngine::Mul>(*mult)) {
            subset2_new.push_back(dim);
            continue;
        }

        // Offset must be a symbol and moving
        if (!SymEngine::is_a<SymEngine::Symbol>(*offset)) {
            subset2_new.push_back(dim);
            continue;
        }
        auto off = SymEngine::rcp_static_cast<const SymEngine::Symbol>(offset);
        if (moving_symbols.find(off->get_name()) == moving_symbols.end()) {
            subset2_new.push_back(dim);
            continue;
        }

        // Multiplier must be two symbols and one moving
        auto mult_ = SymEngine::rcp_static_cast<const SymEngine::Mul>(mult);
        if (mult_->get_args().size() != 2) {
            subset2_new.push_back(dim);
            continue;
        }
        auto multiplier = mult_->get_args()[0];
        auto indvar_ = mult_->get_args()[1];
        if (!SymEngine::is_a<SymEngine::Symbol>(*multiplier) ||
            !SymEngine::is_a<SymEngine::Symbol>(*indvar_)) {
            subset2_new.push_back(dim);
            continue;
        }
        auto mul = SymEngine::rcp_static_cast<const SymEngine::Symbol>(multiplier);
        auto indvar = SymEngine::rcp_static_cast<const SymEngine::Symbol>(indvar_);
        if (moving_symbols.find(mul->get_name()) != moving_symbols.end()) {
            auto tmp = mul;
            mul = indvar;
            indvar = tmp;
        }
        if (moving_symbols.find(mul->get_name()) != moving_symbols.end() ||
            moving_symbols.find(indvar->get_name()) == moving_symbols.end()) {
            subset2_new.push_back(dim);
            continue;
        }

        bool is_nonnegative = false;
        symbolic::ExpressionSet lbs_off;
        symbolic::lower_bounds(off, assumptions, lbs_off);
        for (auto& lb : lbs_off) {
            if (SymEngine::is_a<SymEngine::Integer>(*lb)) {
                auto lb_int = SymEngine::rcp_static_cast<const SymEngine::Integer>(lb);
                if (lb_int->as_int() >= 0) {
                    is_nonnegative = true;
                    break;
                }
            }
        }
        if (!is_nonnegative) {
            subset2_new.push_back(dim);
            continue;
        }

        bool success = false;
        symbolic::ExpressionSet ubs_off;
        symbolic::upper_bounds(off, assumptions, ubs_off);
        for (auto& ub_off : ubs_off) {
            if (symbolic::eq(mul, symbolic::add(ub_off, symbolic::one()))) {
                subset2_new.push_back(indvar);
                subset2_new.push_back(off);

                success = true;
                break;
            }
        }
        if (success) {
            continue;
        }
        subset2_new.push_back(dim);
    }

    return {subset1_new, subset2_new};
};

bool DataParallelismAnalysis::disjoint(const data_flow::Subset& subset1,
                                       const data_flow::Subset& subset2, const std::string& indvar,
                                       const std::unordered_set<std::string>& moving_symbols,
                                       const symbolic::Assumptions& assumptions) {
    if (subset1.size() != subset2.size()) {
        return false;
    }

    codegen::CPPLanguageExtension language_extension;

    // Attempt to substitute complex constant expressions by parameters
    symbolic::ExpressionMap replacements;
    std::vector<std::string> substitions;
    auto [subset1_, subset2_] = DataParallelismAnalysis::substitution(
        subset1, subset2, indvar, moving_symbols, replacements, substitions);

    // Attempt to delinearize subsets
    auto [subset1_2, subset2_2] =
        DataParallelismAnalysis::delinearization(subset1_, subset2_, moving_symbols, assumptions);

    // Overapproximate multiplications with parameters
    data_flow::Subset subset1_new;
    for (auto& dim : subset1_2) {
        auto dim_ = dim;
        for (auto mul : symbolic::muls(dim)) {
            auto mul_ = SymEngine::rcp_static_cast<const SymEngine::Mul>(mul);
            auto arg1 = mul_->get_args()[0];
            if (SymEngine::is_a<SymEngine::Symbol>(*arg1) &&
                moving_symbols.find(
                    SymEngine::rcp_static_cast<const SymEngine::Symbol>(arg1)->get_name()) ==
                    moving_symbols.end()) {
                dim_ = symbolic::subs(dim_, mul_, symbolic::one());
            } else {
                auto arg2 = mul_->get_args()[1];
                if (SymEngine::is_a<SymEngine::Symbol>(*arg2) &&
                    moving_symbols.find(
                        SymEngine::rcp_static_cast<const SymEngine::Symbol>(arg2)->get_name()) ==
                        moving_symbols.end()) {
                    dim_ = symbolic::subs(dim_, mul_, symbolic::one());
                }
            }
        }
        subset1_new.push_back(dim_);
    }
    data_flow::Subset subset2_new;
    for (auto& dim : subset2_2) {
        auto dim_ = dim;
        for (auto mul : symbolic::muls(dim)) {
            auto mul_ = SymEngine::rcp_static_cast<const SymEngine::Mul>(mul);
            auto arg1 = mul_->get_args()[0];
            if (SymEngine::is_a<SymEngine::Symbol>(*arg1) &&
                moving_symbols.find(
                    SymEngine::rcp_static_cast<const SymEngine::Symbol>(arg1)->get_name()) ==
                    moving_symbols.end()) {
                dim_ = symbolic::subs(dim_, mul_, symbolic::one());
            } else {
                auto arg2 = mul_->get_args()[1];
                if (SymEngine::is_a<SymEngine::Symbol>(*arg2) &&
                    moving_symbols.find(
                        SymEngine::rcp_static_cast<const SymEngine::Symbol>(arg2)->get_name()) ==
                        moving_symbols.end()) {
                    dim_ = symbolic::subs(dim_, mul_, symbolic::one());
                }
            }
        }
        subset2_new.push_back(dim_);
    }

    // Collect parameters and dimensions
    std::unordered_set<std::string> dimensions_;
    std::unordered_set<std::string> parameters_;
    for (auto& dim : subset1_new) {
        for (auto& atom : symbolic::atoms(dim)) {
            auto sym = SymEngine::rcp_static_cast<const SymEngine::Symbol>(atom);
            if (sym->get_name() == indvar) {
                continue;
            }

            if (std::find(substitions.begin(), substitions.end(), sym->get_name()) !=
                substitions.end()) {
                continue;
            }
            if (moving_symbols.find(sym->get_name()) == moving_symbols.end()) {
                parameters_.insert(sym->get_name());
            } else {
                dimensions_.insert(sym->get_name());
            }
        }
    }
    for (auto& dim : subset2_new) {
        for (auto& atom : symbolic::atoms(dim)) {
            auto sym = SymEngine::rcp_static_cast<const SymEngine::Symbol>(atom);
            if (sym->get_name() == indvar) {
                continue;
            }

            if (std::find(substitions.begin(), substitions.end(), sym->get_name()) !=
                substitions.end()) {
                continue;
            }
            if (moving_symbols.find(sym->get_name()) == moving_symbols.end()) {
                parameters_.insert(sym->get_name());
            } else {
                dimensions_.insert(sym->get_name());
            }
        }
    }
    dimensions_.insert(indvar);
    std::vector<std::string> dimensions;
    for (auto& dim : dimensions_) {
        dimensions.push_back(dim);
    }
    for (auto mv : moving_symbols) {
        if (dimensions_.find(mv) == dimensions_.end()) {
            dimensions.push_back(mv);
            dimensions_.insert(mv);
        }
    }
    std::vector<std::string> parameters;
    for (auto& dim : parameters_) {
        parameters.push_back(dim);
    }

    // Double dimension space and constraints
    size_t k = 0;
    std::vector<std::string> doubled_dimensions;
    std::vector<std::string> constraints;
    for (auto& dim : dimensions) {
        doubled_dimensions.push_back(dim + "_1");
        doubled_dimensions.push_back(dim + "_2");

        // Proof: dim_1 != dim_2
        if (dim == indvar) {
            constraints.push_back(dim + "_1 != " + dim + "_2");
        }

        // Collect lb and ub
        auto lb1 = assumptions.at(symbolic::symbol(dim)).lower_bound();
        for (auto atom : symbolic::atoms(lb1)) {
            auto sym = SymEngine::rcp_static_cast<const SymEngine::Symbol>(atom);
            if (moving_symbols.find(sym->get_name()) == moving_symbols.end()) {
                if (parameters_.find(sym->get_name()) == parameters_.end()) {
                    parameters_.insert(sym->get_name());
                    parameters.push_back(sym->get_name());
                }
            }
        }
        auto ub1 = assumptions.at(symbolic::symbol(dim)).upper_bound();
        for (auto atom : symbolic::atoms(ub1)) {
            auto sym = SymEngine::rcp_static_cast<const SymEngine::Symbol>(atom);
            if (moving_symbols.find(sym->get_name()) == moving_symbols.end()) {
                if (parameters_.find(sym->get_name()) == parameters_.end()) {
                    parameters_.insert(sym->get_name());
                    parameters.push_back(sym->get_name());
                }
            }
        }

        // Add constraints
        auto lb2 = lb1;
        auto ub2 = ub1;
        for (auto& dim : dimensions) {
            lb1 = symbolic::subs(lb1, symbolic::symbol(dim), symbolic::symbol(dim + "_1"));
            ub1 = symbolic::subs(ub1, symbolic::symbol(dim), symbolic::symbol(dim + "_1"));
            lb2 = symbolic::subs(lb2, symbolic::symbol(dim), symbolic::symbol(dim + "_2"));
            ub2 = symbolic::subs(ub2, symbolic::symbol(dim), symbolic::symbol(dim + "_2"));
        }

        if (!SymEngine::eq(*lb1, *symbolic::infty(-1))) {
            auto mins = SymEngine::atoms<const SymEngine::Min>(*lb1);
            if (mins.size() > 0) {
                continue;
            }

            if (SymEngine::is_a<SymEngine::Max>(*lb1)) {
                auto max = SymEngine::rcp_static_cast<const SymEngine::Max>(lb1);
                auto args1 = max->get_args();
                constraints.push_back(language_extension.expression(args1[0]) + " <= " + dim +
                                      "_1");
                constraints.push_back(language_extension.expression(args1[1]) + " <= " + dim +
                                      "_1");

                auto max_ = SymEngine::rcp_static_cast<const SymEngine::Max>(lb2);
                auto args2 = max_->get_args();
                constraints.push_back(language_extension.expression(args2[0]) + " <= " + dim +
                                      "_2");
                constraints.push_back(language_extension.expression(args2[1]) + " <= " + dim +
                                      "_2");
            } else {
                constraints.push_back(language_extension.expression(lb1) + " <= " + dim + "_1");
                constraints.push_back(language_extension.expression(lb2) + " <= " + dim + "_2");
            }
        }
        if (!SymEngine::eq(*ub1, *symbolic::infty(1))) {
            auto maxs = SymEngine::atoms<const SymEngine::Max>(*ub1);
            if (maxs.size() > 0) {
                continue;
            }

            if (SymEngine::is_a<SymEngine::Min>(*ub1)) {
                auto min = SymEngine::rcp_static_cast<const SymEngine::Min>(ub1);
                auto args1 = min->get_args();
                constraints.push_back(dim + "_1 <= " + language_extension.expression(args1[0]));
                constraints.push_back(dim + "_1 <= " + language_extension.expression(args1[1]));

                auto min_ = SymEngine::rcp_static_cast<const SymEngine::Min>(ub2);
                auto args2 = min_->get_args();
                constraints.push_back(dim + "_2 <= " + language_extension.expression(args2[0]));
                constraints.push_back(dim + "_2 <= " + language_extension.expression(args2[1]));
            } else {
                constraints.push_back(dim + "_1 <= " + language_extension.expression(ub1));
                constraints.push_back(dim + "_2 <= " + language_extension.expression(ub2));
            }
        }

        // Add map constraints
        auto map = assumptions.at(symbolic::symbol(dim)).map();
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
        if (!SymEngine::is_a<SymEngine::Integer>(*lb1)) {
            continue;
        }

        std::string new_k = "__daisy_iterator" + std::to_string(k++);

        std::string k_1 = new_k + "_1";
        constraints.push_back("exists " + k_1 + " : " + dim +
                              "_1 = " + language_extension.expression(lb1) + " + " + k_1 + " * " +
                              language_extension.expression(arg1));

        std::string k_2 = new_k + "_2";
        constraints.push_back("exists " + k_2 + " : " + dim +
                              "_2 = " + language_extension.expression(lb1) + " + " + k_2 + " * " +
                              language_extension.expression(arg1));
    }

    // Extend parameters by dependening parameters
    size_t num_params = parameters.size();
    size_t num_params_old = 0;
    do {
        for (size_t i = 0; i < num_params; i++) {
            auto param = parameters[i];
            auto lb = assumptions.at(symbolic::symbol(param)).lower_bound();
            for (auto& atom : symbolic::atoms(lb)) {
                auto sym = SymEngine::rcp_static_cast<const SymEngine::Symbol>(atom);
                if (moving_symbols.find(sym->get_name()) == moving_symbols.end()) {
                    if (parameters_.find(sym->get_name()) == parameters_.end()) {
                        parameters_.insert(sym->get_name());
                        parameters.push_back(sym->get_name());
                    }
                }
            }
            auto ub = assumptions.at(symbolic::symbol(param)).upper_bound();
            for (auto& atom : symbolic::atoms(ub)) {
                auto sym = SymEngine::rcp_static_cast<const SymEngine::Symbol>(atom);
                if (moving_symbols.find(sym->get_name()) == moving_symbols.end()) {
                    if (parameters_.find(sym->get_name()) == parameters_.end()) {
                        parameters_.insert(sym->get_name());
                        parameters.push_back(sym->get_name());
                    }
                }
            }
        }
        num_params_old = num_params;
        num_params = parameters.size();
    } while (num_params != num_params_old);

    // Collect constraints for parameters
    for (size_t i = 0; i < parameters.size(); i++) {
        auto lb = assumptions.at(symbolic::symbol(parameters[i])).lower_bound();
        auto ub = assumptions.at(symbolic::symbol(parameters[i])).upper_bound();

        std::string constraint = "";
        if (!SymEngine::eq(*lb, *symbolic::infty(-1))) {
            if (SymEngine::is_a<SymEngine::Max>(*lb)) {
                auto max = SymEngine::rcp_static_cast<const SymEngine::Max>(lb);
                auto args = max->get_args();
                constraints.push_back(language_extension.expression(args[0]) +
                                      " <= " + parameters[i]);
                constraints.push_back(language_extension.expression(args[1]) +
                                      " <= " + parameters[i]);
            } else if (SymEngine::atoms<const SymEngine::Min>(*lb).size() > 0) {
            } else {
                constraints.push_back(language_extension.expression(lb) + " <= " + parameters[i]);
            }
        }
        if (!SymEngine::eq(*ub, *symbolic::infty(1))) {
            if (SymEngine::is_a<SymEngine::Min>(*ub)) {
                auto min = SymEngine::rcp_static_cast<const SymEngine::Min>(ub);
                auto args = min->get_args();
                constraints.push_back(parameters[i] +
                                      " <= " + language_extension.expression(args[0]));
                constraints.push_back(parameters[i] +
                                      " <= " + language_extension.expression(args[1]));
            } else if (SymEngine::atoms<const SymEngine::Max>(*ub).size() > 0) {
            } else {
                constraints.push_back(parameters[i] + " <= " + language_extension.expression(ub));
            }
        }
    }

    // Allocate context
    isl_ctx* ctx = isl_ctx_alloc();

    // Define maps
    std::string map_1;
    if (!parameters.empty()) {
        map_1 += "[";
        map_1 += helpers::join(parameters, ", ");
    }
    if (!substitions.empty()) {
        if (!parameters.empty()) {
            map_1 += ", ";
        } else {
            map_1 += "[";
        }
        map_1 += helpers::join(substitions, ", ");
    }
    if (!map_1.empty()) {
        map_1 += "] -> ";
    }
    map_1 += "{ [" + helpers::join(doubled_dimensions, ", ") + "] -> [";
    for (size_t i = 0; i < subset1_new.size(); i++) {
        auto dim = subset1_new[i];
        for (auto& iter : dimensions) {
            dim = symbolic::subs(dim, symbolic::symbol(iter), symbolic::symbol(iter + "_1"));
        }
        map_1 += language_extension.expression(dim);
        if (i < subset1_new.size() - 1) {
            map_1 += ", ";
        }
    }
    map_1 += "] : " + helpers::join(constraints, " and ") + " }";

    std::string map_2;
    if (!parameters.empty()) {
        map_2 += "[";
        map_2 += helpers::join(parameters, ", ");
    }
    if (!substitions.empty()) {
        if (!parameters.empty()) {
            map_2 += ", ";
        } else {
            map_2 += "[";
        }
        map_2 += helpers::join(substitions, ", ");
    }
    if (!map_2.empty()) {
        map_2 += "] -> ";
    }
    map_2 += "{ [" + helpers::join(doubled_dimensions, ", ") + "] -> [";
    for (size_t i = 0; i < subset2_new.size(); i++) {
        auto dim = subset2_new[i];
        for (auto& iter : dimensions) {
            dim = symbolic::subs(dim, symbolic::symbol(iter), symbolic::symbol(iter + "_2"));
        }
        map_2 += language_extension.expression(dim);
        if (i < subset2_new.size() - 1) {
            map_2 += ", ";
        }
    }
    map_2 += "] : " + helpers::join(constraints, " and ") + " }";

    // Replace NV symbols with names without .
    map_1 = std::regex_replace(map_1, std::regex("\\."), "_");
    map_2 = std::regex_replace(map_2, std::regex("\\."), "_");

    isl_map* index_map_1 = isl_map_read_from_str(ctx, map_1.c_str());
    if (!index_map_1) {
        isl_ctx_free(ctx);
        return false;
    }

    isl_map* index_map_2 = isl_map_read_from_str(ctx, map_2.c_str());
    if (!index_map_2) {
        isl_map_free(index_map_1);
        isl_ctx_free(ctx);
        return false;
    }

    isl_map* intersection = isl_map_intersect(index_map_1, index_map_2);
    if (!intersection) {
        isl_map_free(index_map_1);
        isl_map_free(index_map_2);
        isl_ctx_free(ctx);
        return false;
    }

    bool disjoint = isl_map_is_empty(intersection);

    isl_map_free(intersection);
    isl_ctx_free(ctx);

    return disjoint;
};

void DataParallelismAnalysis::classify(analysis::AnalysisManager& analysis_manager,
                                       structured_control_flow::StructuredLoop* loop) {
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();

    // Strictly monotonic update
    auto& indvar = loop->indvar();
    if (!loop_analysis.is_monotonic(loop)) {
        this->results_.insert({loop, DataParallelismAnalysisResult()});
        return;
    }

    // Users analysis
    auto& body = loop->root();
    auto& users = analysis_manager.get<analysis::Users>();
    analysis::UsersView body_users(users, body);
    if (!body_users.views().empty() || !body_users.moves().empty()) {
        this->results_.insert({loop, DataParallelismAnalysisResult()});
        return;
    }

    this->results_.insert({loop, DataParallelismAnalysisResult()});
    auto& result = this->results_.at(loop);

    // Assumptions analysis
    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto assumptions = assumptions_analysis.get(body, true);

    // For each container, we now classify the access pattern

    // 1. Identify private containers
    auto locals = users.locals(body);
    for (auto& local : locals) {
        result.insert({local, Parallelism::PRIVATE});
    }

    // 2. Filter our read-only containers
    std::unordered_set<std::string> writeset;
    std::unordered_set<std::string> moving_symbols = {indvar->get_name()};
    for (auto& entry : body_users.writes()) {
        writeset.insert(entry->container());

        auto& type = sdfg_.type(entry->container());
        if (!dynamic_cast<const types::Scalar*>(&type)) {
            continue;
        }
        if (!types::is_integer(type.primitive_type())) {
            continue;
        }
        moving_symbols.insert(entry->container());
    }
    for (auto& entry : body_users.reads()) {
        if (writeset.find(entry->container()) != writeset.end()) {
            // Loop-carried dependencies in tasklets's conditions
            if (dynamic_cast<data_flow::Tasklet*>(entry->element())) {
                // Locals cannot be loop-carried
                if (locals.find(entry->container()) != locals.end()) {
                    continue;
                } else {
                    result.clear();
                    return;
                }
            }

            continue;
        }
        if (entry->container() == indvar->get_name()) {
            continue;
        }
        result.insert({entry->container(), Parallelism::READONLY});
    }

    // 3. Prove parallelism
    /** For each container of the writeset, we check the following properties:
     *  1. For all i, writes are disjoint (false -> empty parallelism)
     *  2. For all i, reads and writes are disjoint (false -> empty parallelism)
     */
    for (auto& container : writeset) {
        // Skip if already classified
        if (result.find(container) != result.end()) {
            continue;
        }

        // For each i1, i2, subset(i1) and subset(i2) are disjoint
        bool ww_conflict = false;
        auto writes = body_users.writes(container);
        for (auto& write : writes) {
            for (auto& write_ : writes) {
                if (write == write_ && writes.size() > 1) {
                    continue;
                }

                // Determine moving symbols locally
                std::unordered_set<std::string> moving_symbols_local = moving_symbols;

                auto subsets = write->subsets();
                auto subsets_ = write_->subsets();
                for (auto& subset : subsets) {
                    for (auto& subset_ : subsets_) {
                        if (!this->disjoint(subset, subset_, indvar->get_name(),
                                            moving_symbols_local, assumptions)) {
                            ww_conflict = true;
                            break;
                        }
                    }
                    if (ww_conflict) {
                        break;
                    }
                }
                if (ww_conflict) {
                    break;
                }
            }
            if (ww_conflict) {
                break;
            }
        }
        if (ww_conflict) {
            result.insert({container, Parallelism::DEPENDENT});
            continue;
        }

        bool rw_conflict = false;
        auto reads = body_users.reads(container);
        for (auto& read : reads) {
            for (auto& write_ : writes) {
                // Determine moving symbols locally
                std::unordered_set<std::string> moving_symbols_local = moving_symbols;

                auto subsets = read->subsets();
                auto subsets_ = write_->subsets();
                for (auto& subset : subsets) {
                    for (auto& subset_ : subsets_) {
                        if (!this->disjoint(subset, subset_, indvar->get_name(),
                                            moving_symbols_local, assumptions)) {
                            rw_conflict = true;
                            break;
                        }
                    }
                    if (rw_conflict) {
                        break;
                    }
                }
                if (rw_conflict) {
                    break;
                }
            }
            if (rw_conflict) {
                break;
            }
        }
        if (rw_conflict) {
            result.insert({container, Parallelism::DEPENDENT});
            continue;
        }

        result.insert({container, Parallelism::PARALLEL});
    }

    // 4. Reductions
    for (auto& entry : result) {
        auto& container = entry.first;
        auto& dep_type = entry.second;
        if (dep_type != Parallelism::DEPENDENT) {
            continue;
        }

        // Check if it is a reduction
        auto reads = body_users.reads(container);
        auto writes = body_users.writes(container);

        // Criterion: Must write to constant location
        bool is_reduction = true;
        auto first_write = writes.at(0);
        auto first_subset = first_write->subsets().at(0);
        for (auto& dim : first_subset) {
            for (auto& sym : moving_symbols) {
                if (symbolic::uses(dim, sym)) {
                    is_reduction = false;
                    break;
                }
            }
            if (!is_reduction) {
                break;
            }
        }
        if (!is_reduction) {
            continue;
        }

        // Criterion: All writes must have the same subset
        for (auto& write : writes) {
            for (auto& subset : write->subsets()) {
                if (subset.size() != first_subset.size()) {
                    is_reduction = false;
                    break;
                }

                for (size_t i = 0; i < subset.size(); i++) {
                    if (!symbolic::eq(subset[i], first_subset[i])) {
                        is_reduction = false;
                        break;
                    }
                }
            }
        }
        if (!is_reduction) {
            continue;
        }

        // Criterion: All reads must have the same subset
        for (auto& read : reads) {
            for (auto& subset : read->subsets()) {
                if (subset.size() != first_subset.size()) {
                    is_reduction = false;
                    break;
                }

                for (size_t i = 0; i < subset.size(); i++) {
                    if (!symbolic::eq(subset[i], first_subset[i])) {
                        is_reduction = false;
                        break;
                    }
                }
            }
        }

        if (is_reduction) {
            result[container] = Parallelism::REDUCTION;
        }
    }
};

void DataParallelismAnalysis::run(analysis::AnalysisManager& analysis_manager) {
    this->results_.clear();

    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    for (auto& loop : loop_analysis.loops()) {
        if (auto sloop = dynamic_cast<structured_control_flow::StructuredLoop*>(loop)) {
            this->classify(analysis_manager, sloop);
        }
    }
};

DataParallelismAnalysis::DataParallelismAnalysis(StructuredSDFG& sdfg)
    : Analysis(sdfg) {

      };

const DataParallelismAnalysisResult& DataParallelismAnalysis::get(
    const structured_control_flow::StructuredLoop& loop) const {
    return this->results_.at(&loop);
};

symbolic::Expression DataParallelismAnalysis::bound(
    const structured_control_flow::StructuredLoop& loop) {
    auto& indvar = loop.indvar();
    auto& condition = loop.condition();
    auto args = condition->get_args();
    if (args.size() != 2) {
        return SymEngine::null;
    }
    auto& arg0 = args[0];
    auto& arg1 = args[1];
    if (SymEngine::eq(*arg0, *indvar)) {
        return arg1;
    } else if (SymEngine::eq(*arg1, *indvar)) {
        return arg0;
    }
    return SymEngine::null;
};

}  // namespace analysis
}  // namespace sdfg