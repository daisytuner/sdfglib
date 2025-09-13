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

namespace types {
// forward declaration, because it depends on contents of this file
class IType;
} // namespace types

namespace symbolic {

/**** Basic Definitions ****/
typedef SymEngine::RCP<const SymEngine::Symbol> Symbol;
typedef SymEngine::RCP<const SymEngine::Number> Number;
typedef SymEngine::RCP<const SymEngine::Integer> Integer;
typedef SymEngine::RCP<const SymEngine::Infty> Infty;
typedef SymEngine::RCP<const SymEngine::Basic> Expression;
typedef SymEngine::RCP<const SymEngine::Boolean> Condition;

typedef std::vector<Expression> MultiExpression;

// Datastructures
typedef std::vector<Symbol> SymbolVec;
typedef std::set<Symbol, SymEngine::RCPBasicKeyLess> SymbolSet;
typedef std::unordered_map<Symbol, Expression, SymEngine::RCPBasicHash, SymEngine::RCPBasicKeyEq> SymbolMap;

typedef std::vector<Symbol> ExpressionVec;
typedef std::set<Expression, SymEngine::RCPBasicKeyLess> ExpressionSet;
typedef std::unordered_map<Expression, Expression, SymEngine::RCPBasicHash, SymEngine::RCPBasicKeyEq> ExpressionMap;

/***** Special Symbols *****/

Symbol symbol(const std::string& name);

Integer integer(int64_t value);

Integer zero();

Integer one();

Condition __false__();

Condition __true__();

Symbol __nullptr__();

bool is_nullptr(const Symbol symbol);

bool is_pointer(const Symbol symbol);

bool is_nv(const Symbol symbol);

/***** Logical Expressions *****/

Condition And(const Condition lhs, const Condition rhs);

Condition Or(const Condition lhs, const Condition rhs);

Condition Not(const Condition expr);

bool is_true(const Expression expr);

bool is_false(const Expression expr);

/***** Integer Functions *****/

Expression add(const Expression lhs, const Expression rhs);

Expression sub(const Expression lhs, const Expression rhs);

Expression mul(const Expression lhs, const Expression rhs);

Expression div(const Expression lhs, const Expression rhs);

Expression min(const Expression lhs, const Expression rhs);

Expression max(const Expression lhs, const Expression rhs);

Expression mod(const Expression lhs, const Expression rhs);

Expression pow(const Expression base, const Expression exp);

Expression size_of_type(const types::IType& type);

class SizeOfTypeFunction : public SymEngine::FunctionSymbol {
private:
    const types::IType& type_;

public:
    explicit SizeOfTypeFunction(const types::IType& type)
        : FunctionSymbol("sizeof", SymEngine::vec_basic{}), type_(type) {}

    const types::IType& get_type() const { return type_; }
};

/***** Comparisions *****/

Condition Eq(const Expression lhs, const Expression rhs);

Condition Ne(const Expression lhs, const Expression rhs);

Condition Lt(const Expression lhs, const Expression rhs);

Condition Gt(const Expression lhs, const Expression rhs);

Condition Le(const Expression lhs, const Expression rhs);

Condition Ge(const Expression lhs, const Expression rhs);

/***** Modification *****/

Expression expand(const Expression expr);

Expression simplify(const Expression expr);

bool eq(const Expression lhs, const Expression rhs);

bool uses(const Expression expr, const Symbol sym);

bool uses(const Expression expr, const std::string& name);

SymbolSet atoms(const Expression expr);

ExpressionSet muls(const Expression expr);

Expression subs(const Expression expr, const Expression old_expr, const Expression new_expr);

Condition subs(const Condition expr, const Expression old_expr, const Expression new_expr);

/***** NV Symbols *****/

Symbol threadIdx_x();

Symbol threadIdx_y();

Symbol threadIdx_z();

Symbol blockDim_x();

Symbol blockDim_y();

Symbol blockDim_z();

Symbol blockIdx_x();

Symbol blockIdx_y();

Symbol blockIdx_z();

Symbol gridDim_x();

Symbol gridDim_y();

Symbol gridDim_z();

Expression parse(const std::string& expr_str);

} // namespace symbolic
} // namespace sdfg
