#include "sdfg/symbolic/conjunctive_normal_form.h"

#include <gtest/gtest.h>

using namespace sdfg;

TEST(ConjunctiveNormalFormTest, Basic) {
    auto a = symbolic::symbol("a");
    auto b = symbolic::symbol("b");

    auto atom_a = symbolic::Eq(a, symbolic::integer(1));
    auto atom_b = symbolic::Le(b, symbolic::integer(2));

    auto expr = symbolic::Or(symbolic::And(atom_a, atom_b),
                             symbolic::And(symbolic::Not(atom_a), symbolic::Not(atom_b)));

    auto cnf = symbolic::conjunctive_normal_form(expr);
    EXPECT_EQ(cnf.size(), 4);
}
