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
CNF conjunctive_normal_form(const Condition cond);

} // namespace symbolic
} // namespace sdfg
