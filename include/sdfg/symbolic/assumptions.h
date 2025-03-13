#pragma once

#include <memory>

#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace symbolic {

class Assumption {
   private:
    Symbol symbol_;

    // Domain
    Expression lower_bound_;
    Expression upper_bound_;
    Expression map_;

   public:
    Assumption();

    Assumption(const Symbol& symbol);

    Assumption(const Assumption& a);

    Assumption& operator=(const Assumption& a);

    const Symbol& symbol() const;

    const Expression& lower_bound() const;

    void lower_bound(const Expression& lower_bound);

    const Expression& upper_bound() const;

    void upper_bound(const Expression& upper_bound);

    const Integer integer_value() const;

    const Expression& map() const;

    void map(const Expression& map);

    bool is_positive() const;

    bool is_negative() const;

    bool is_nonnegative() const;

    bool is_nonpositive() const;

    static Assumption create(const symbolic::Symbol& symbol, const types::IType& type);
};

typedef std::unordered_map<Symbol, Assumption, SymEngine::RCPBasicHash, SymEngine::RCPBasicKeyEq>
    Assumptions;

void upper_bounds(const symbolic::Symbol& sym, const Assumptions& assumptions,
                  symbolic::SymbolicSet& ubs);

void lower_bounds(const symbolic::Symbol& sym, const Assumptions& assumptions,
                  symbolic::SymbolicSet& lbs);

}  // namespace symbolic
}  // namespace sdfg
