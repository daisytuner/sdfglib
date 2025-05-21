#pragma once

#include <symengine/add.h>
#include <symengine/basic.h>
#include <symengine/dict.h>
#include <symengine/integer.h>
#include <symengine/logic.h>
#include <symengine/mul.h>
#include <symengine/nan.h>
#include <symengine/real_double.h>
#include <symengine/simplify.h>
#include <symengine/symbol.h>

#include <unordered_map>

namespace sdfg {

namespace symbolic {

// Atoms
typedef SymEngine::RCP<const SymEngine::Symbol> Symbol;
typedef SymEngine::RCP<const SymEngine::Integer> Integer;
typedef SymEngine::RCP<const SymEngine::Infty> Infty;

// Expressions
typedef SymEngine::RCP<const SymEngine::Basic> Expression;
typedef SymEngine::RCP<const SymEngine::Boolean> Condition;

// Datastructures
typedef std::unordered_map<Symbol, Expression, SymEngine::RCPBasicHash, SymEngine::RCPBasicKeyEq>
    Assignments;
typedef std::vector<SymEngine::RCP<const SymEngine::Symbol>> SymbolicVector;
typedef std::set<SymEngine::RCP<const SymEngine::Basic>, SymEngine::RCPBasicKeyLess> SymbolicSet;
typedef std::unordered_map<Expression, Expression, SymEngine::RCPBasicHash,
                           SymEngine::RCPBasicKeyEq>
    SymbolicMap;

/***** Special Symbols *****/

Symbol symbol(const std::string& name);

Integer integer(int64_t value);

Integer zero();

Integer one();

Infty infty(int direction);

Condition __false__();

Condition __true__();

Symbol __nullptr__();

bool is_nullptr(const Symbol& symbol);

bool is_memory_address(const Symbol& symbol);

bool is_pointer(const Symbol& symbol);

bool is_nvptx(const Symbol& symbol);

/***** Logical Expressions *****/

Condition And(const Condition& lhs, const Condition& rhs);

Condition Or(const Condition& lhs, const Condition& rhs);

Condition Not(const Condition& expr);

bool is_true(const Expression& expr);

bool is_false(const Expression& expr);

/***** Arithmetic Expressions *****/

Expression add(const Expression& lhs, const Expression& rhs);

Expression sub(const Expression& lhs, const Expression& rhs);

Expression mul(const Expression& lhs, const Expression& rhs);

Expression div(const Expression& lhs, const Expression& rhs);

Expression min(const Expression& lhs, const Expression& rhs);

Expression max(const Expression& lhs, const Expression& rhs);

/***** Trigonometric Expressions *****/

Expression sin(const Expression& expr);

Expression cos(const Expression& expr);

Expression tan(const Expression& expr);

Expression cot(const Expression& expr);

/***** Logarithmic Expressions  *****/

Expression log(const Expression& expr);

/***** Power Expressions  *****/

Expression pow(const Expression& base, const Expression& exp);

Expression exp(const Expression& expr);

Expression sqrt(const Expression& expr);

/***** Comparisions *****/

Condition Eq(const Expression& lhs, const Expression& rhs);

Condition Ne(const Expression& lhs, const Expression& rhs);

Condition Lt(const Expression& lhs, const Expression& rhs);

Condition Gt(const Expression& lhs, const Expression& rhs);

Condition Le(const Expression& lhs, const Expression& rhs);

Condition Ge(const Expression& lhs, const Expression& rhs);

/***** Modification *****/

bool eq(const Expression& lhs, const Expression& rhs);

bool uses(const Expression& expr, const Symbol& sym);

bool uses(const Expression& expr, const std::string& name);

SymbolicSet atoms(const Expression& expr);

SymbolicSet muls(const Expression& expr);

Expression subs(const Expression& expr, const Expression& old_expr, const Expression& new_expr);

Condition subs(const Condition& expr, const Expression& old_expr, const Expression& new_expr);

Condition simplify(const Condition& expr);

Expression simplify(const Expression& expr);

}  // namespace symbolic
}  // namespace sdfg
