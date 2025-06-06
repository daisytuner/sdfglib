#include "sdfg/symbolic/symbolic.h"

#include <string>

#include "sdfg/exceptions.h"
#include "symengine/functions.h"
#include "symengine/logic.h"

namespace sdfg {
namespace symbolic {

Symbol symbol(const std::string& name) {
    if (name == "null") {
        throw InvalidSDFGException("null is not a valid symbol");
    } else if (name == "NULL") {
        throw InvalidSDFGException("NULL is not a valid symbol");
    } else if (name == "nullptr") {
        throw InvalidSDFGException("nullptr is not a valid symbol");
    }

    return SymEngine::symbol(name);
};

Integer integer(int64_t value) { return SymEngine::integer(value); };

Integer zero() { return symbolic::integer(0); };

Integer one() { return symbolic::integer(1); };

Infty infty(int direction) { return SymEngine::infty(direction); };

Condition __false__() { return SymEngine::boolean(false); };

Condition __true__() { return SymEngine::boolean(true); };

Symbol __nullptr__() { return SymEngine::symbol("__daisy_nullptr"); };

bool is_memory_address(const Symbol& symbol) {
    return symbol->get_name().starts_with("reinterpret_cast");
};

bool is_nullptr(const Symbol& symbol) { return symbol->get_name() == "__daisy_nullptr"; };

bool is_pointer(const Symbol& symbol) { return is_memory_address(symbol) || is_nullptr(symbol); };

bool is_nv(const Symbol& symbol) {
    if (symbol == threadIdx_x() || symbol == threadIdx_y() || symbol == threadIdx_z() ||
        symbol == blockIdx_x() || symbol == blockIdx_y() || symbol == blockIdx_z() ||
        symbol == blockDim_x() || symbol == blockDim_y() || symbol == blockDim_z() ||
        symbol == gridDim_x() || symbol == gridDim_y() || symbol == gridDim_z()) {
        return true;
    } else {
        return false;
    }
};

/***** Logical Expressions *****/

Condition And(const Condition& lhs, const Condition& rhs) {
    return SymEngine::logical_and({lhs, rhs});
};

Condition Or(const Condition& lhs, const Condition& rhs) {
    return SymEngine::logical_or({lhs, rhs});
};

Condition Not(const Condition& expr) { return expr->logical_not(); };

bool is_true(const Expression& expr) { return SymEngine::eq(*SymEngine::boolTrue, *expr); };

bool is_false(const Expression& expr) { return SymEngine::eq(*SymEngine::boolFalse, *expr); };

/***** Arithmetic Expressions *****/

Expression add(const Expression& lhs, const Expression& rhs) { return SymEngine::add(lhs, rhs); };

Expression sub(const Expression& lhs, const Expression& rhs) { return SymEngine::sub(lhs, rhs); };

Expression mul(const Expression& lhs, const Expression& rhs) { return SymEngine::mul(lhs, rhs); };

Expression div(const Expression& lhs, const Expression& rhs) { return SymEngine::div(lhs, rhs); };

Expression min(const Expression& lhs, const Expression& rhs) { return SymEngine::min({lhs, rhs}); };

Expression max(const Expression& lhs, const Expression& rhs) { return SymEngine::max({lhs, rhs}); };

/***** Trigonometric Expressions *****/

Expression sin(const Expression& expr) { return SymEngine::sin(expr); };

Expression cos(const Expression& expr) { return SymEngine::cos(expr); };

Expression tan(const Expression& expr) { return SymEngine::tan(expr); };

Expression cot(const Expression& expr) { return SymEngine::cot(expr); };

/***** Logarithmic Expressions  *****/

Expression log(const Expression& expr) { return SymEngine::log(expr); };

/***** Power Expressions  *****/

Expression pow(const Expression& base, const Expression& exp) { return SymEngine::pow(base, exp); };

Expression exp(const Expression& expr) { return SymEngine::exp(expr); };

Expression sqrt(const Expression& expr) { return SymEngine::sqrt(expr); };

/***** Comparisions *****/

Condition Eq(const Expression& lhs, const Expression& rhs) { return SymEngine::Eq(lhs, rhs); };

Condition Ne(const Expression& lhs, const Expression& rhs) { return SymEngine::Ne(lhs, rhs); };

Condition Lt(const Expression& lhs, const Expression& rhs) { return SymEngine::Lt(lhs, rhs); };

Condition Gt(const Expression& lhs, const Expression& rhs) { return SymEngine::Gt(lhs, rhs); };

Condition Le(const Expression& lhs, const Expression& rhs) { return SymEngine::Le(lhs, rhs); };

Condition Ge(const Expression& lhs, const Expression& rhs) { return SymEngine::Ge(lhs, rhs); };

/***** Rounding Expressions *****/

Expression floor(const Expression& expr) { return SymEngine::floor(expr); };

Expression ceil(const Expression& expr) { return SymEngine::ceiling(expr); };

/***** Modification *****/

bool eq(const Expression& lhs, const Expression& rhs) { return SymEngine::eq(*lhs, *rhs); };

bool uses(const Expression& expr, const Symbol& sym) { return SymEngine::has_symbol(*expr, *sym); };

bool uses(const Expression& expr, const std::string& name) {
    return symbolic::uses(expr, symbol(name));
};

SymbolicSet atoms(const Expression& expr) {
    return SymEngine::atoms<const SymEngine::Symbol>(*expr);
};

SymbolicSet muls(const Expression& expr) { return SymEngine::atoms<const SymEngine::Mul>(*expr); };

Expression subs(const Expression& expr, const Expression& old_expr, const Expression& new_expr) {
    SymEngine::map_basic_basic d;
    d[old_expr] = new_expr;

    return expr->subs(d);
};

Condition subs(const Condition& expr, const Expression& old_expr, const Expression& new_expr) {
    SymEngine::map_basic_basic d;
    d[old_expr] = new_expr;

    return SymEngine::rcp_static_cast<const SymEngine::Boolean>(expr->subs(d));
};

Condition simplify(const Condition& expr) {
    auto true_expr = symbolic::__true__();
    auto false_expr = symbolic::__false__();
    if (SymEngine::is_a<SymEngine::Equality>(*expr)) {
        auto args = expr->get_args();
        if (!SymEngine::is_a_Boolean(*args[0]) || !SymEngine::is_a_Boolean(*args[1])) {
            return expr;
        }
        auto arg0 = SymEngine::rcp_dynamic_cast<const SymEngine::Boolean>(args[0]);
        auto arg1 = SymEngine::rcp_dynamic_cast<const SymEngine::Boolean>(args[1]);

        if (SymEngine::eq(*arg0, *false_expr)) {
            return symbolic::Not(arg1);
        } else if (SymEngine::eq(*arg1, *false_expr)) {
            return symbolic::Not(arg0);
        } else if (SymEngine::eq(*arg0, *true_expr)) {
            return arg1;
        } else if (SymEngine::eq(*arg1, *true_expr)) {
            return arg0;
        }
    } else if (SymEngine::is_a<SymEngine::Unequality>(*expr)) {
        auto args = expr->get_args();
        if (!SymEngine::is_a_Boolean(*args[0]) || !SymEngine::is_a_Boolean(*args[1])) {
            return expr;
        }
        auto arg0 = SymEngine::rcp_dynamic_cast<const SymEngine::Boolean>(args[0]);
        auto arg1 = SymEngine::rcp_dynamic_cast<const SymEngine::Boolean>(args[1]);

        if (SymEngine::eq(*arg0, *false_expr)) {
            return arg1;
        } else if (SymEngine::eq(*arg1, *false_expr)) {
            return arg0;
        } else if (SymEngine::eq(*arg0, *true_expr)) {
            return symbolic::Not(arg1);
        } else if (SymEngine::eq(*arg1, *true_expr)) {
            return symbolic::Not(arg0);
        }
    } else if (SymEngine::is_a<SymEngine::And>(*expr)) {
        auto elements = expr->get_args();
        std::vector<Condition> simplified;
        for (auto& element : elements) {
            bool found = false;
            if (SymEngine::is_a<SymEngine::LessThan>(*element)) {
                auto args = element->get_args();
                for (auto& mirror : elements) {
                    if (SymEngine::is_a<SymEngine::LessThan>(*mirror)) {
                        auto mirror_args = mirror->get_args();
                        if (symbolic::eq(args[0], mirror_args[1]) &&
                            symbolic::eq(args[1], mirror_args[0])) {
                            auto arg0 = args[0];
                            auto arg1 = args[1];
                            simplified.push_back(symbolic::Eq(arg0, arg1));
                            found = true;
                            continue;
                        }
                    }
                }
            }
            if (!found) {
                simplified.push_back(
                    SymEngine::rcp_dynamic_cast<const SymEngine::Boolean>(element));
            }
        }
        Condition result = symbolic::__true__();
        for (auto& element : simplified) {
            result = symbolic::And(result, element);
        }
        return result;
    }
    return expr;
};

Expression simplify(const Expression& expr) {
    if (SymEngine::is_a_Boolean(*expr)) {
        return symbolic::simplify(SymEngine::rcp_static_cast<const SymEngine::Boolean>(expr));
    } else if (SymEngine::is_a<SymEngine::Min>(*expr)) {
        auto args = expr->get_args();
        if (symbolic::eq(args[0], args[1])) {
            return symbolic::simplify(args[0]);
        } else if (symbolic::eq(args[0], symbolic::infty(1))) {
            return symbolic::simplify(args[1]);
        } else if (symbolic::eq(args[1], symbolic::infty(1))) {
            return symbolic::simplify(args[0]);
        }
        return symbolic::min(symbolic::simplify(args[0]), symbolic::simplify(args[1]));
    } else if (SymEngine::is_a<SymEngine::Max>(*expr)) {
        auto args = expr->get_args();
        if (symbolic::eq(args[0], args[1])) {
            return symbolic::simplify(args[0]);
        } else if (symbolic::eq(args[0], symbolic::infty(-1))) {
            return symbolic::simplify(args[1]);
        } else if (symbolic::eq(args[1], symbolic::infty(-1))) {
            return symbolic::simplify(args[0]);
        }
    }

    return SymEngine::simplify(expr);
};

Condition rearrange_simple_condition(const Condition& inequality, const Symbol& target_symbol) {
    Expression lhs;
    Expression rhs;

    // Check if the inequality is a StrictLessThan
    if (SymEngine::is_a<SymEngine::StrictLessThan>(*inequality)) {
        auto lt_expr = SymEngine::rcp_dynamic_cast<const SymEngine::StrictLessThan>(inequality);
        lhs = lt_expr->get_arg1();
        rhs = lt_expr->get_arg2();
    } else if (SymEngine::is_a<SymEngine::LessThan>(*inequality)) {
        auto lt_expr = SymEngine::rcp_dynamic_cast<const SymEngine::LessThan>(inequality);
        lhs = lt_expr->get_arg1();
        rhs = lt_expr->get_arg2();
    } else if (SymEngine::is_a<SymEngine::Equality>(*inequality)) {
        auto gt_expr = SymEngine::rcp_dynamic_cast<const SymEngine::Equality>(inequality);
        lhs = gt_expr->get_arg1();
        rhs = gt_expr->get_arg2();
    } else {
        return inequality;  // Return original if not a strict less than
    }

    // Check if the target_symbol is part of an Add on the LHS
    if (SymEngine::is_a<SymEngine::Add>(*lhs)) {
        Expression add_expr = SymEngine::rcp_static_cast<const SymEngine::Add>(lhs);

        bool term_with_exactly_one_symbol = false;

        for (auto& term : add_expr->get_args()) {
            if (SymEngine::is_a<SymEngine::Symbol>(*term)) {
                auto symbol_term = SymEngine::rcp_dynamic_cast<const SymEngine::Symbol>(term);
                if (eq(symbol_term, target_symbol) && !term_with_exactly_one_symbol) {
                    term_with_exactly_one_symbol = true;
                } else if (eq(symbol_term, target_symbol) && term_with_exactly_one_symbol) {
                    return inequality;  // More than one symbol in the term
                }
            }
        }

        if (SymEngine::is_a<SymEngine::StrictLessThan>(*inequality)) {
            return symbolic::Lt(target_symbol,
                                symbolic::sub(rhs, subs(lhs, target_symbol, zero())));
        } else if (SymEngine::is_a<SymEngine::LessThan>(*inequality)) {
            return symbolic::Le(target_symbol,
                                symbolic::sub(rhs, subs(lhs, target_symbol, zero())));
        } else if (SymEngine::is_a<SymEngine::Equality>(*inequality)) {
            return symbolic::Eq(target_symbol,
                                symbolic::sub(rhs, subs(lhs, target_symbol, zero())));
        }
    }
    // More complex cases (multiplication, division, other functions) would need more logic here
    // e.g., if lhs is Mul(2, i) < N => i < N/2
    //       if lhs is Mul(-1, i) < N => i > -N (reverse inequality)

    return inequality;  // Return original if rearrangement is not handled
}

/***** NV Symbols *****/

Symbol threadIdx_x() { return symbol("threadIdx.x"); };

Symbol threadIdx_y() { return symbol("threadIdx.y"); };

Symbol threadIdx_z() { return symbol("threadIdx.z"); };

Symbol blockDim_x() { return symbol("blockDim.x"); };

Symbol blockDim_y() { return symbol("blockDim.y"); };

Symbol blockDim_z() { return symbol("blockDim.z"); };

Symbol blockIdx_x() { return symbol("blockIdx.x"); };

Symbol blockIdx_y() { return symbol("blockIdx.y"); };

Symbol blockIdx_z() { return symbol("blockIdx.z"); };

Symbol gridDim_x() { return symbol("gridDim.x"); };

Symbol gridDim_y() { return symbol("gridDim.y"); };

Symbol gridDim_z() { return symbol("gridDim.z"); };

}  // namespace symbolic
}  // namespace sdfg
