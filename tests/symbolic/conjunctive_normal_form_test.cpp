#include "sdfg/symbolic/conjunctive_normal_form.h"

#include <gtest/gtest.h>

using namespace sdfg;

TEST(ConjunctiveNormalFormTest, Literal_Eq_Integral) {
    auto a = symbolic::symbol("a");

    auto literal = symbolic::Eq(a, symbolic::integer(1));
    auto cnf = symbolic::conjunctive_normal_form(literal);
    EXPECT_EQ(cnf.size(), 1);
    EXPECT_EQ(cnf[0].size(), 1);
    EXPECT_TRUE(symbolic::eq(cnf[0][0], literal));
}

TEST(ConjunctiveNormalFormTest, Literal_Eq_False) {
    auto a = symbolic::symbol("a");

    auto literal = symbolic::Eq(a, symbolic::__false__());
    auto cnf = symbolic::conjunctive_normal_form(literal);
    EXPECT_EQ(cnf.size(), 1);
    EXPECT_EQ(cnf[0].size(), 1);
    EXPECT_TRUE(symbolic::eq(cnf[0][0], literal));
}

TEST(ConjunctiveNormalFormTest, Literal_Eq_True) {
    auto a = symbolic::symbol("a");

    auto literal = symbolic::Eq(a, symbolic::__true__());
    auto cnf = symbolic::conjunctive_normal_form(literal);
    EXPECT_EQ(cnf.size(), 1);
    EXPECT_EQ(cnf[0].size(), 1);
    EXPECT_TRUE(symbolic::eq(cnf[0][0], literal));
}

TEST(ConjunctiveNormalFormTest, Literal_Simplify_Eq_False) {
    auto a = symbolic::symbol("a");

    auto literal = symbolic::Eq(symbolic::Eq(a, symbolic::integer(1)), symbolic::__false__());
    auto cnf = symbolic::conjunctive_normal_form(literal);
    EXPECT_EQ(cnf.size(), 1);
    EXPECT_EQ(cnf[0].size(), 1);
    EXPECT_TRUE(symbolic::eq(cnf[0][0], symbolic::Ne(a, symbolic::integer(1))));
}

TEST(ConjunctiveNormalFormTest, Literal_Simplify_Eq_True) {
    auto a = symbolic::symbol("a");

    auto literal = symbolic::Eq(symbolic::Eq(a, symbolic::integer(1)), symbolic::__true__());
    auto cnf = symbolic::conjunctive_normal_form(literal);
    EXPECT_EQ(cnf.size(), 1);
    EXPECT_EQ(cnf[0].size(), 1);
    EXPECT_TRUE(symbolic::eq(cnf[0][0], symbolic::Eq(a, symbolic::integer(1))));
}

TEST(ConjunctiveNormalFormTest, Literal_Simplify_Ne_False) {
    auto a = symbolic::symbol("a");

    auto literal = symbolic::Ne(symbolic::Eq(a, symbolic::integer(1)), symbolic::__false__());
    auto cnf = symbolic::conjunctive_normal_form(literal);
    EXPECT_EQ(cnf.size(), 1);
    EXPECT_EQ(cnf[0].size(), 1);
    EXPECT_TRUE(symbolic::eq(cnf[0][0], symbolic::Eq(a, symbolic::integer(1))));
}

TEST(ConjunctiveNormalFormTest, Literal_Simplify_Ne_True) {
    auto a = symbolic::symbol("a");

    auto literal = symbolic::Ne(symbolic::Eq(a, symbolic::integer(1)), symbolic::__true__());
    auto cnf = symbolic::conjunctive_normal_form(literal);
    EXPECT_EQ(cnf.size(), 1);
    EXPECT_EQ(cnf[0].size(), 1);
    EXPECT_TRUE(symbolic::eq(cnf[0][0], symbolic::Ne(a, symbolic::integer(1))));
}

TEST(ConjunctiveNormalFormTest, Basic) {
    auto a = symbolic::symbol("a");
    auto b = symbolic::symbol("b");

    auto atom_a = symbolic::Eq(a, symbolic::integer(1));
    auto atom_b = symbolic::Le(b, symbolic::integer(2));

    auto expr =
        symbolic::Or(symbolic::And(atom_a, atom_b), symbolic::And(symbolic::Not(atom_a), symbolic::Not(atom_b)));

    auto cnf = symbolic::conjunctive_normal_form(expr);
    EXPECT_EQ(cnf.size(), 4);
}
