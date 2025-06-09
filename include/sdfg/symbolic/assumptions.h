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

    static Assumption create(const symbolic::Symbol& symbol, const types::IType& type);
};

typedef std::unordered_map<Symbol, Assumption, SymEngine::RCPBasicHash, SymEngine::RCPBasicKeyEq> Assumptions;

}  // namespace symbolic
}  // namespace sdfg
