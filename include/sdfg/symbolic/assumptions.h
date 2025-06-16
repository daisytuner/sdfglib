#pragma once

#include <unordered_map>

#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace symbolic {

class Assumption {
   private:
    Symbol symbol_;
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

    const Expression& map() const;

    void map(const Expression& map);
};

typedef std::unordered_map<Symbol, Assumption, SymEngine::RCPBasicHash, SymEngine::RCPBasicKeyEq>
    Assumptions;

void upper_bounds(const symbolic::Symbol& sym, const Assumptions& assumptions,
                  symbolic::ExpressionSet& ubs);

void lower_bounds(const symbolic::Symbol& sym, const Assumptions& assumptions,
                  symbolic::ExpressionSet& lbs);

}  // namespace symbolic
}  // namespace sdfg
