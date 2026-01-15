#include "sdfg/symbolic/conjunctive_normal_form.h"

#include <symengine/logic.h>

#include "sdfg/symbolic/extreme_values.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace symbolic {

CNF distribute_or(const CNF& C, const CNF& D) {
    CNF out;
    for (auto& c : C)
        for (auto& d : D) {
            auto clause = c;
            clause.insert(clause.end(), d.begin(), d.end());
            out.emplace_back(std::move(clause));
        }
    return out;
}

CNF conjunctive_normal_form(const Condition cond) {
    // Goal: Convert a condition into ANDs of ORs

    // Case: Comparison with boolean literals
    if (SymEngine::is_a<SymEngine::Equality>(*cond) || SymEngine::is_a<SymEngine::Unequality>(*cond)) {
        auto expr = SymEngine::rcp_static_cast<const SymEngine::Relational>(cond);
        auto arg1 = expr->get_arg1();
        auto arg2 = expr->get_arg2();
        if (!SymEngine::is_a_Relational(*arg1) && !SymEngine::is_a_Relational(*arg2)) {
            return {{cond}};
        }

        if (SymEngine::is_a<SymEngine::Equality>(*expr)) {
            if (symbolic::is_true(arg1)) {
                return conjunctive_normal_form(SymEngine::rcp_static_cast<const SymEngine::Boolean>(arg2));
            } else if (symbolic::is_true(arg2)) {
                return conjunctive_normal_form(SymEngine::rcp_static_cast<const SymEngine::Boolean>(arg1));
            } else if (symbolic::is_false(arg1)) {
                return conjunctive_normal_form(symbolic::Not(SymEngine::rcp_static_cast<const SymEngine::Boolean>(arg2))
                );
            } else if (symbolic::is_false(arg2)) {
                return conjunctive_normal_form(symbolic::Not(SymEngine::rcp_static_cast<const SymEngine::Boolean>(arg1))
                );
            }
        } else if (SymEngine::is_a<SymEngine::Unequality>(*expr)) {
            if (symbolic::is_true(arg1)) {
                return conjunctive_normal_form(symbolic::Not(SymEngine::rcp_static_cast<const SymEngine::Boolean>(arg2))
                );
            } else if (symbolic::is_true(arg2)) {
                return conjunctive_normal_form(symbolic::Not(SymEngine::rcp_static_cast<const SymEngine::Boolean>(arg1))
                );
            } else if (symbolic::is_false(arg1)) {
                return conjunctive_normal_form(SymEngine::rcp_static_cast<const SymEngine::Boolean>(arg2));
            } else if (symbolic::is_false(arg2)) {
                return conjunctive_normal_form(SymEngine::rcp_static_cast<const SymEngine::Boolean>(arg1));
            }
        }

        return {{cond}}; // Return the condition as a single clause
    }

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
        CNF result;
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

        CNF result;
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

} // namespace symbolic
} // namespace sdfg
