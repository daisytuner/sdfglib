#include "sdfg/symbolic/symbolic.h"

#include <string>

#include <symengine/parser.h>
#include "sdfg/exceptions.h"
#include "sdfg/types/type.h"
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

bool is_nullptr(const Symbol& symbol) { return symbol->get_name() == "__daisy_nullptr"; };

bool is_pointer(const Symbol& symbol) { return is_nullptr(symbol); };

bool is_nv(const Symbol& symbol) {
    if (symbol == threadIdx_x() || symbol == threadIdx_y() || symbol == threadIdx_z() || symbol == blockIdx_x() ||
        symbol == blockIdx_y() || symbol == blockIdx_z() || symbol == blockDim_x() || symbol == blockDim_y() ||
        symbol == blockDim_z() || symbol == gridDim_x() || symbol == gridDim_y() || symbol == gridDim_z()) {
        return true;
    } else {
        return false;
    }
};

/***** Logical Expressions *****/

Condition And(const Condition& lhs, const Condition& rhs) { return SymEngine::logical_and({lhs, rhs}); };

Condition Or(const Condition& lhs, const Condition& rhs) { return SymEngine::logical_or({lhs, rhs}); };

Condition Not(const Condition& expr) { return expr->logical_not(); };

bool is_true(const Expression& expr) { return SymEngine::eq(*SymEngine::boolTrue, *expr); };

bool is_false(const Expression& expr) { return SymEngine::eq(*SymEngine::boolFalse, *expr); };

/***** Integer Functions *****/

Expression add(const Expression& lhs, const Expression& rhs) { return SymEngine::add(lhs, rhs); };

Expression sub(const Expression& lhs, const Expression& rhs) { return SymEngine::sub(lhs, rhs); };

Expression mul(const Expression& lhs, const Expression& rhs) { return SymEngine::mul(lhs, rhs); };

Expression div(const Expression& lhs, const Expression& rhs) {
    if (eq(rhs, integer(1))) {
        return lhs;
    } else if (SymEngine::is_a<SymEngine::Integer>(*lhs) && SymEngine::is_a<SymEngine::Integer>(*rhs)) {
        auto a = SymEngine::rcp_static_cast<const SymEngine::Integer>(lhs)->as_int();
        auto b = SymEngine::rcp_static_cast<const SymEngine::Integer>(rhs)->as_int();
        return integer(a / b);
    }
    auto idiv = SymEngine::function_symbol("idiv", {lhs, rhs});
    return idiv;
};

Expression min(const Expression& lhs, const Expression& rhs) { return SymEngine::min({lhs, rhs}); };

Expression max(const Expression& lhs, const Expression& rhs) { return SymEngine::max({lhs, rhs}); };

Expression mod(const Expression& lhs, const Expression& rhs) {
    if (SymEngine::is_a<SymEngine::Integer>(*lhs) && SymEngine::is_a<SymEngine::Integer>(*rhs)) {
        auto a = SymEngine::rcp_static_cast<const SymEngine::Integer>(lhs)->as_int();
        auto b = SymEngine::rcp_static_cast<const SymEngine::Integer>(rhs)->as_int();
        return integer(a % b);
    }
    auto imod = SymEngine::function_symbol("imod", {lhs, rhs});
    return imod;
};

Expression pow(const Expression& base, const Expression& exp) { return SymEngine::pow(base, exp); };

Expression size_of_type(const types::IType& type) {
    auto so = SymEngine::make_rcp<SizeOfTypeFunction>(type);
    return so;
}

/***** Comparisions *****/

Condition Eq(const Expression& lhs, const Expression& rhs) { return SymEngine::Eq(lhs, rhs); };

Condition Ne(const Expression& lhs, const Expression& rhs) { return SymEngine::Ne(lhs, rhs); };

Condition Lt(const Expression& lhs, const Expression& rhs) { return SymEngine::Lt(lhs, rhs); };

Condition Gt(const Expression& lhs, const Expression& rhs) { return SymEngine::Gt(lhs, rhs); };

Condition Le(const Expression& lhs, const Expression& rhs) { return SymEngine::Le(lhs, rhs); };

Condition Ge(const Expression& lhs, const Expression& rhs) { return SymEngine::Ge(lhs, rhs); };

/***** Modification *****/

Expression expand(const Expression& expr) {
    auto new_expr = SymEngine::expand(expr);
    return new_expr;
};

Expression simplify(const Expression& expr) {
    if (SymEngine::is_a<SymEngine::FunctionSymbol>(*expr)) {
        auto func_sym = SymEngine::rcp_static_cast<const SymEngine::FunctionSymbol>(expr);
        auto func_id = func_sym->get_name();
        if (func_id == "idiv") {
            auto lhs = func_sym->get_args()[0];
            auto rhs = func_sym->get_args()[1];
            if (SymEngine::is_a<SymEngine::Mul>(*lhs) && SymEngine::is_a<SymEngine::Integer>(*rhs)) {
                auto lhs_mul = SymEngine::rcp_static_cast<const SymEngine::Mul>(lhs);
                auto rhs_int = SymEngine::rcp_static_cast<const SymEngine::Integer>(rhs);
                auto lhs_args = lhs_mul->get_args();

                bool skipped = false;
                Expression new_mul = SymEngine::integer(1);
                for (auto& arg : lhs_args) {
                    if (eq(arg, rhs_int) && !skipped) {
                        skipped = true;
                    } else {
                        new_mul = SymEngine::mul(new_mul, arg);
                    }
                }
                if (skipped) {
                    return new_mul;
                }
            } else if (SymEngine::is_a<SymEngine::Integer>(*lhs) && SymEngine::is_a<SymEngine::Integer>(*rhs)) {
                auto a = SymEngine::rcp_static_cast<const SymEngine::Integer>(lhs)->as_int();
                auto b = SymEngine::rcp_static_cast<const SymEngine::Integer>(rhs)->as_int();
                return integer(a / b);
            }
        } else if (func_id == "imod") {
            auto lhs = func_sym->get_args()[0];
            auto rhs = func_sym->get_args()[1];
            if (SymEngine::is_a<SymEngine::Integer>(*lhs) && SymEngine::is_a<SymEngine::Integer>(*rhs)) {
                auto a = SymEngine::rcp_static_cast<const SymEngine::Integer>(lhs)->as_int();
                auto b = SymEngine::rcp_static_cast<const SymEngine::Integer>(rhs)->as_int();
                return integer(a % b);
            }
        }
    }

    try {
        auto new_expr = SymEngine::simplify(expr);
        return new_expr;
    } catch (const SymEngine::SymEngineException& e) {
        std::cout << "Error simplifying expression: " << e.what() << std::endl;
        return expr;
    }
};

bool eq(const Expression& lhs, const Expression& rhs) { return SymEngine::eq(*lhs, *rhs); };

bool uses(const Expression& expr, const Symbol& sym) { return SymEngine::has_symbol(*expr, *sym); };

bool uses(const Expression& expr, const std::string& name) { return symbolic::uses(expr, symbol(name)); };

SymbolSet atoms(const Expression& expr) {
    SymbolSet atoms;
    for (auto& atom : SymEngine::atoms<const SymEngine::Basic>(*expr)) {
        if (SymEngine::is_a<SymEngine::Symbol>(*atom)) {
            atoms.insert(SymEngine::rcp_static_cast<const SymEngine::Symbol>(atom));
        }
    }
    return atoms;
};

ExpressionSet muls(const Expression& expr) { return SymEngine::atoms<const SymEngine::Mul>(*expr); };

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

Expression parse(const std::string& expr_str) { return SymEngine::parse(expr_str); };

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

} // namespace symbolic
} // namespace sdfg
