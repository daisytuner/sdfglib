#pragma once

#include <unordered_map>

#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace symbolic {

class Assumption {
private:
    Symbol symbol_;
    Expression lower_bound_deprecated_;
    Expression upper_bound_deprecated_;
    ExpressionSet lower_bounds_;
    ExpressionSet upper_bounds_;
    Expression tight_lower_bound_;
    Expression tight_upper_bound_;
    bool constant_;
    Expression map_;

public:
    Assumption();

    Assumption(const Symbol symbol);

    Assumption(const Assumption& a);

    Assumption& operator=(const Assumption& a);

    const Symbol symbol() const;

    //[[deprecated("use lower_bound/tight_lower_bound instead")]]
    const Expression lower_bound_deprecated() const;

    //[[deprecated("use lower_bound/tight_lower_bound instead")]]
    void lower_bound_deprecated(const Expression lower_bound);

    //[[deprecated("use upper_bound/tight_upper_bound instead")]]
    const Expression upper_bound_deprecated() const;

    //[[deprecated("use upper_bound/tight_upper_bound instead")]]
    void upper_bound_deprecated(const Expression upper_bound);

    const Expression lower_bound() const;

    const ExpressionSet& lower_bounds() const;

    void add_lower_bound(const Expression lb);

    bool contains_lower_bound(const Expression lb);

    bool remove_lower_bound(const Expression lb);

    const Expression upper_bound() const;

    const ExpressionSet& upper_bounds() const;

    void add_upper_bound(const Expression ub);

    bool contains_upper_bound(const Expression ub);

    bool remove_upper_bound(const Expression ub);

    const Expression tight_lower_bound() const;

    void tight_lower_bound(const Expression tight_lb);

    const Expression tight_upper_bound() const;

    void tight_upper_bound(const Expression tight_ub);

    bool constant() const;

    void constant(bool constant);

    const Expression map() const;

    void map(const Expression map);

    static Assumption create(const Symbol symbol, const types::IType& type);
};

typedef std::unordered_map<Symbol, Assumption, SymEngine::RCPBasicHash, SymEngine::RCPBasicKeyEq> Assumptions;

} // namespace symbolic
} // namespace sdfg
