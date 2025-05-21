#include "sdfg/symbolic/analysis.h"

#include "sdfg/symbolic/symbolic.h"
#include "symengine/infinity.h"
#include "symengine/polys/basic_conversions.h"
#include "symengine/visitor.h"

namespace sdfg {
namespace symbolic {

Polynomial polynomial(const Expression& expr, const Symbol& symbol) {
    try {
        return SymEngine::from_basic<SymEngine::UExprPoly>(expr, symbol, true);
    } catch (SymEngine::SymEngineException& e) {
        return SymEngine::null;
    }
};

MultiPolynomial multi_polynomial(const Expression& expr, SymbolicVector& symbols) {
    try {
        SymbolicSet gens;
        for (auto& symbol : symbols) {
            gens.insert(symbol);
        }
        return SymEngine::from_basic<SymEngine::MExprPoly>(expr, gens);
    } catch (SymEngine::SymEngineException& e) {
        return SymEngine::null;
    }
};

AffineCoefficients affine_coefficients(MultiPolynomial& poly, SymbolicVector& symbols) {
    AffineCoefficients coefficients;
    for (auto& symbol : symbols) {
        coefficients[symbol] = 0;
    }
    coefficients[symbolic::symbol("__daisy_constant__")] = 0;

    auto& D = poly->get_poly().get_dict();
    for (auto& [exponents, coeff] : D) {
        // Check if coeff is an integer
        if (!SymEngine::is_a<SymEngine::Integer>(coeff)) {
            return {};
        }
        auto coeff_value =
            SymEngine::rcp_dynamic_cast<const SymEngine::Integer>(coeff.get_basic())->as_int();

        // Check if sum of exponents is <= 1
        symbolic::Symbol hot_symbol = symbolic::symbol("__daisy_constant__");
        unsigned total_deg = 0;
        for (size_t i = 0; i < exponents.size(); i++) {
            auto& e = exponents[i];
            if (e > 0) {
                hot_symbol = symbols[i];
            }
            total_deg += e;
        }
        if (total_deg > 1) {
            return {};
        }

        // Add coefficient to corresponding symbol
        coefficients[hot_symbol] = coefficients[hot_symbol] + coeff_value;
    }

    return coefficients;
}

std::vector<std::vector<Condition>> distribute_or(const std::vector<std::vector<Condition>>& C,
                                                  const std::vector<std::vector<Condition>>& D) {
    std::vector<std::vector<Condition>> out;
    for (auto& c : C)
        for (auto& d : D) {
            auto clause = c;
            clause.insert(clause.end(), d.begin(), d.end());
            out.emplace_back(std::move(clause));
        }
    return out;
}

std::vector<std::vector<Condition>> conjunctive_normal_form(const Condition& cond) {
    // Goal: Convert a condition into ANDs of ORs

    // Case: Not
    // Push negation inwards
    if (SymEngine::is_a<SymEngine::Not>(*cond)) {
        auto not_ = SymEngine::rcp_static_cast<const SymEngine::Not>(cond);
        auto arg = SymEngine::rcp_dynamic_cast<const SymEngine::Boolean>(not_->get_arg());

        // Case: Not(not)
        if (SymEngine::is_a<SymEngine::Not>(*arg)) {
            auto not_not_ = SymEngine::rcp_static_cast<const SymEngine::Not>(arg);
            auto arg_ = SymEngine::rcp_dynamic_cast<const SymEngine::Boolean>(not_not_->get_arg());
            return conjunctive_normal_form(arg_);
        }

        // Case: Not(And) (De Morgan)
        if (SymEngine::is_a<SymEngine::And>(*arg)) {
            auto and_ = SymEngine::rcp_static_cast<const SymEngine::And>(arg);
            auto args = and_->get_args();
            if (args.size() != 2) {
                throw CNFException("Non-binary And encountered");
            }
            auto arg0 = SymEngine::rcp_dynamic_cast<const SymEngine::Boolean>(args[0]);
            auto arg1 = SymEngine::rcp_dynamic_cast<const SymEngine::Boolean>(args[1]);
            auto de_morgan = symbolic::Or(symbolic::Not(arg0), symbolic::Not(arg1));
            return conjunctive_normal_form(de_morgan);
        }

        // Case: Not(Or) (De Morgan)
        if (SymEngine::is_a<SymEngine::Or>(*arg)) {
            auto or_ = SymEngine::rcp_static_cast<const SymEngine::Or>(arg);
            auto args = or_->get_args();
            if (args.size() != 2) {
                throw CNFException("Non-binary Or encountered");
            }
            auto arg0 = SymEngine::rcp_dynamic_cast<const SymEngine::Boolean>(args[0]);
            auto arg1 = SymEngine::rcp_dynamic_cast<const SymEngine::Boolean>(args[1]);
            auto de_morgan = symbolic::And(symbolic::Not(arg0), symbolic::Not(arg1));
            return conjunctive_normal_form(de_morgan);
        }

        // Case: Comparisons
        if (SymEngine::is_a<SymEngine::Equality>(*arg)) {
            auto eq_ = SymEngine::rcp_static_cast<const SymEngine::Equality>(arg);
            auto lhs = eq_->get_arg1();
            auto rhs = eq_->get_arg2();
            return conjunctive_normal_form(symbolic::Ne(lhs, rhs));
        }
        if (SymEngine::is_a<SymEngine::Unequality>(*arg)) {
            auto ne_ = SymEngine::rcp_static_cast<const SymEngine::Unequality>(arg);
            auto lhs = ne_->get_arg1();
            auto rhs = ne_->get_arg2();
            return conjunctive_normal_form(symbolic::Eq(lhs, rhs));
        }
        if (SymEngine::is_a<SymEngine::LessThan>(*arg)) {
            auto lt_ = SymEngine::rcp_static_cast<const SymEngine::LessThan>(arg);
            auto lhs = lt_->get_arg1();
            auto rhs = lt_->get_arg2();
            return conjunctive_normal_form(symbolic::Gt(lhs, rhs));
        }
        if (SymEngine::is_a<SymEngine::StrictLessThan>(*arg)) {
            auto lt_ = SymEngine::rcp_static_cast<const SymEngine::StrictLessThan>(arg);
            auto lhs = lt_->get_arg1();
            auto rhs = lt_->get_arg2();
            return conjunctive_normal_form(symbolic::Ge(lhs, rhs));
        }

        throw CNFException("Unknown Not encountered");
    }

    // Case: And
    if (SymEngine::is_a<SymEngine::And>(*cond)) {
        // CNF(A ∧ B) = CNF(A)  ∪  CNF(B)
        auto and_ = SymEngine::rcp_static_cast<const SymEngine::And>(cond);
        auto args = and_->get_args();
        std::vector<std::vector<Condition>> result;
        for (auto& arg : args) {
            auto arg_ = SymEngine::rcp_dynamic_cast<const SymEngine::Boolean>(arg);
            auto cnf = conjunctive_normal_form(arg_);
            for (auto& clause : cnf) {
                result.push_back(clause);
            }
        }
        return result;
    }

    // Case: Or
    if (SymEngine::is_a<SymEngine::Or>(*cond)) {
        // CNF(A ∨ B) = distribute_or( CNF(A), CNF(B) )
        auto or_ = SymEngine::rcp_static_cast<const SymEngine::Or>(cond);
        auto args = or_->get_args();

        std::vector<std::vector<Condition>> result;
        auto arg_0 = SymEngine::rcp_dynamic_cast<const SymEngine::Boolean>(args[0]);
        auto cnf_0 = conjunctive_normal_form(arg_0);
        for (auto& clause : cnf_0) {
            result.push_back(clause);
        }
        for (size_t i = 1; i < args.size(); i++) {
            auto arg = args[i];
            auto arg_ = SymEngine::rcp_dynamic_cast<const SymEngine::Boolean>(arg);
            auto cnf = conjunctive_normal_form(arg_);
            result = distribute_or(result, cnf);
        }
        return result;
    }

    // Case: Literal
    return {{cond}};
}

Affine affine(const Expression& expr, const Symbol& symbol) {
    auto poly = symbolic::polynomial(expr, symbol);
    if (poly == SymEngine::null || poly->get_degree() > 1) {
        return {SymEngine::null, SymEngine::null};
    }
    if (poly->get_degree() == 0) {
        return {symbolic::integer(0), poly->get_coeff(0)};
    }
    return {poly->get_coeff(1), poly->get_coeff(0)};
};

Sign strict_monotonicity_affine(const Expression& func, const Symbol& symbol) {
    return strict_monotonicity_affine(func, symbol, {});
};

Sign strict_monotonicity_affine(const Expression& func, const symbolic::Assumptions& assumptions) {
    Sign sign = Sign::NONE;
    bool initialized = false;

    for (auto atom : atoms(func)) {
        auto sym = SymEngine::rcp_static_cast<const SymEngine::Symbol>(atom);
        if (!initialized) {
            sign = strict_monotonicity_affine(func, sym, assumptions);
            if (sign == Sign::NONE) {
                return Sign::NONE;
            }
            initialized = true;
            continue;
        }
        if (sign != strict_monotonicity_affine(func, sym, assumptions)) {
            return Sign::NONE;
        }
    }
    return sign;
};

Sign strict_monotonicity_affine(const Expression& func, const Symbol& symbol,
                                const symbolic::Assumptions& assumptions) {
    auto aff = symbolic::affine(func, symbol);
    if (aff.first == SymEngine::null) {
        return Sign::NONE;
    }

    if (SymEngine::is_a<SymEngine::Integer>(*aff.first)) {
        auto coeff = SymEngine::rcp_static_cast<const SymEngine::Integer>(aff.first);
        if (coeff->as_int() > 0) {
            return Sign::POSITIVE;
        } else if (coeff->as_int() < 0) {
            return Sign::NEGATIVE;
        } else {
            return Sign::NONE;
        }
    } else if (SymEngine::is_a<SymEngine::Symbol>(*aff.first)) {
        auto sym = SymEngine::rcp_static_cast<const SymEngine::Symbol>(aff.first);
        if (assumptions.find(sym) != assumptions.end()) {
            auto& a = assumptions.at(sym);
            if (a.is_positive()) {
                return Sign::POSITIVE;
            } else if (a.is_negative()) {
                return Sign::NEGATIVE;
            } else {
                return Sign::NONE;
            }
        }
    }

    return Sign::NONE;
};

Sign strict_monotonicity(const Expression& func, const Symbol& symbol) {
    return strict_monotonicity(func, symbol, {});
};

Sign strict_monotonicity(const Expression& func, const symbolic::Assumptions& assumptions) {
    if (symbolic::strict_monotonicity_affine(func, assumptions) != Sign::NONE) {
        return symbolic::strict_monotonicity_affine(func, assumptions);
    }
    return Sign::NONE;
};

Sign strict_monotonicity(const Expression& func, const Symbol& symbol,
                         const symbolic::Assumptions& assumptions) {
    if (symbolic::strict_monotonicity_affine(func, symbol, assumptions) != Sign::NONE) {
        return symbolic::strict_monotonicity_affine(func, symbol, assumptions);
    }

    return Sign::NONE;
};

Sign strict_monotonicity(const Expression& func) {
    for (auto& sym : symbolic::atoms(func)) {
        auto sym_ = SymEngine::rcp_static_cast<const SymEngine::Symbol>(sym);
        if (symbolic::strict_monotonicity(func, sym_) != Sign::POSITIVE) {
            return Sign::NONE;
        }
    }
    return Sign::POSITIVE;
};

bool contiguity_affine(const Expression& func, const Symbol& symbol) {
    return contiguity_affine(func, symbol, {});
};

bool contiguity_affine(const Expression& func, const Symbol& symbol,
                       const symbolic::Assumptions& assumptions) {
    auto aff = symbolic::affine(func, symbol);
    if (aff.first == SymEngine::null) {
        return false;
    }

    if (SymEngine::is_a<SymEngine::Integer>(*aff.first)) {
        auto coeff = SymEngine::rcp_static_cast<const SymEngine::Integer>(aff.first);
        if (coeff->as_int() == 1) {
            return true;
        }
    } else if (SymEngine::is_a<SymEngine::Symbol>(*aff.first)) {
        auto sym = SymEngine::rcp_static_cast<const SymEngine::Symbol>(aff.first);
        if (assumptions.find(sym) != assumptions.end()) {
            auto& a = assumptions.at(sym);
            if (symbolic::eq(a.lower_bound(), symbolic::integer(1)) &&
                symbolic::eq(a.upper_bound(), symbolic::integer(1))) {
                return true;
            }
        }
    }

    return false;
};

bool contiguity(const Expression& func, const Symbol& symbol) {
    return contiguity(func, symbol, {});
}

bool contiguity(const Expression& func, const Symbol& symbol,
                const symbolic::Assumptions& assumptions) {
    if (symbolic::contiguity_affine(func, symbol, assumptions)) {
        return true;
    }

    return false;
};

Expression inverse(const Symbol& lhs, const Expression& rhs) {
    if (!symbolic::uses(rhs, lhs)) {
        return SymEngine::null;
    }

    if (SymEngine::is_a<SymEngine::Add>(*rhs)) {
        auto add = SymEngine::rcp_static_cast<const SymEngine::Add>(rhs);
        auto arg_0 = add->get_args()[0];
        auto arg_1 = add->get_args()[1];
        if (!symbolic::eq(arg_0, lhs)) {
            std::swap(arg_0, arg_1);
        }
        if (!symbolic::eq(arg_0, lhs)) {
            return SymEngine::null;
        }
        if (!SymEngine::is_a<SymEngine::Integer>(*arg_1)) {
            return SymEngine::null;
        }
        return symbolic::sub(lhs, arg_1);
    } else if (SymEngine::is_a<SymEngine::Mul>(*rhs)) {
        auto mul = SymEngine::rcp_static_cast<const SymEngine::Mul>(rhs);
        auto arg_0 = mul->get_args()[0];
        auto arg_1 = mul->get_args()[1];
        if (!symbolic::eq(arg_0, lhs)) {
            std::swap(arg_0, arg_1);
        }
        if (!symbolic::eq(arg_0, lhs)) {
            return SymEngine::null;
        }
        if (!SymEngine::is_a<SymEngine::Integer>(*arg_1)) {
            return SymEngine::null;
        }
        return symbolic::div(lhs, arg_1);
    }

    return SymEngine::null;
};

bool contains_infinity(const Expression& expr) {
    if (SymEngine::atoms<const SymEngine::Infty>(*expr).empty()) {
        return false;
    }
    return true;
};

Expression lower_bound_analysis(const symbolic::Expression& expr,
                                symbolic::Assumptions& assumptions) {
    if (SymEngine::is_a<SymEngine::Integer>(*expr)) {
        return SymEngine::rcp_static_cast<const SymEngine::Integer>(expr);
    }
    if (SymEngine::is_a<SymEngine::Symbol>(*expr)) {
        auto symbol = SymEngine::rcp_static_cast<const SymEngine::Symbol>(expr);
        if (assumptions.find(symbol) != assumptions.end()) {
            return assumptions[symbol].lower_bound();
        }
        return symbolic::infty(-1);
    }

    // Attempt analysis via monotonicity
    auto sign = symbolic::strict_monotonicity(expr, assumptions);
    if (sign == symbolic::Sign::NONE) {
        return symbolic::infty(-1);
    }

    // Subsitute all symbols accordingly
    symbolic::Expression lower_bound = expr;
    for (auto atom : symbolic::atoms(expr)) {
        auto symbol = SymEngine::rcp_static_cast<const SymEngine::Symbol>(atom);
        if (sign == symbolic::Sign::NEGATIVE) {
            auto ub = assumptions[symbol].upper_bound();
            lower_bound = symbolic::subs(lower_bound, symbol, ub);
        } else {  // if (sign == symbolic::Sign::POSITIVE)
            auto lb = assumptions[symbol].lower_bound();
            lower_bound = symbolic::subs(lower_bound, symbol, lb);
        }
    }

    // End of recursion
    if (contains_infinity(lower_bound)) {
        return symbolic::infty(-1);
    } else {
        if (symbolic::atoms(lower_bound).empty()) {
            return lower_bound;
        } else {
            auto new_lb = lower_bound_analysis(lower_bound, assumptions);
            return new_lb;
        }
    }
};

Expression upper_bound_analysis(const symbolic::Expression& expr,
                                symbolic::Assumptions& assumptions) {
    if (SymEngine::is_a<SymEngine::Integer>(*expr)) {
        return SymEngine::rcp_static_cast<const SymEngine::Integer>(expr);
    }
    if (SymEngine::is_a<SymEngine::Symbol>(*expr)) {
        auto symbol = SymEngine::rcp_static_cast<const SymEngine::Symbol>(expr);
        if (assumptions.find(symbol) != assumptions.end()) {
            return assumptions[symbol].upper_bound();
        }
        return symbolic::infty(1);
    }

    // Attempt analysis via monotonicity
    auto sign = symbolic::strict_monotonicity(expr, assumptions);
    if (sign == symbolic::Sign::NONE) {
        return symbolic::infty(1);
    }

    // Subsitute all symbols accordingly
    symbolic::Expression upper_bound = expr;
    for (auto atom : symbolic::atoms(expr)) {
        auto symbol = SymEngine::rcp_static_cast<const SymEngine::Symbol>(atom);
        if (sign == symbolic::Sign::NEGATIVE) {
            auto lb = assumptions[symbol].lower_bound();
            upper_bound = symbolic::subs(upper_bound, symbol, lb);
        } else {  // if (sign == symbolic::Sign::POSITIVE)
            auto ub = assumptions[symbol].upper_bound();
            upper_bound = symbolic::subs(upper_bound, symbol, ub);
        }
    }

    // End of recursion
    if (contains_infinity(upper_bound)) {
        return symbolic::infty(1);
    } else {
        if (symbolic::atoms(upper_bound).empty()) {
            return upper_bound;
        } else {
            auto new_ub = upper_bound_analysis(upper_bound, assumptions);
            return new_ub;
        }
    }
};
}  // namespace symbolic
}  // namespace sdfg
