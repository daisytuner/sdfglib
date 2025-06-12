#pragma once

#include <vector>

#include "sdfg/symbolic/assumptions.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace symbolic {

class CNFException : public std::exception {
   public:
    CNFException(const std::string& message) : message_(message) {}
    const char* what() const noexcept override { return message_.c_str(); }

   private:
    std::string message_;
};

typedef std::vector<std::vector<Condition>> CNF;

/**
 * @brief Convert a condition to conjunctive normal form.
 *
 * @param cond The condition to convert.
 * @return The conjunctive normal form of the condition.
 */
CNF conjunctive_normal_form(const Condition& cond);

/**
 * @brief Compute an upper bound for a symbol from a conjunctive normal form.
 *
 * The bound may not be tight.
 *
 * @param cnf The conjunctive normal form.
 * @param sym The symbol to derive a bound for.
 * @return The closed form of the conjunctive normal form, SymEngine::null if the CNF is not
 *         closed.
 */
Expression upper_bound(const CNF& cnf, const Symbol& sym);

}  // namespace symbolic
}  // namespace sdfg
