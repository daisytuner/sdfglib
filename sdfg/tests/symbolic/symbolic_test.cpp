#include "sdfg/symbolic/symbolic.h"

#include <gtest/gtest.h>

using namespace sdfg;

TEST(SymbolicTest, Symbols) {
    auto x = symbolic::symbol("x");
    EXPECT_EQ(x->get_name(), "x");
}

TEST(SymbolicTest, Integers) {
    auto num = symbolic::integer(-2);
    EXPECT_EQ(num->as_int(), -2);

    num = symbolic::integer(2);
    EXPECT_EQ(num->as_int(), 2);

    auto zero = symbolic::zero();
    EXPECT_EQ(zero->as_int(), 0);

    auto one = symbolic::one();
    EXPECT_EQ(one->as_int(), 1);

    auto infty = SymEngine::Inf;
    EXPECT_TRUE(SymEngine::eq(*infty, *SymEngine::infty(1)));

    infty = SymEngine::NegInf;
    EXPECT_TRUE(SymEngine::eq(*infty, *SymEngine::infty(-1)));
}

TEST(SymbolicTest, Booleans) {
    auto boolean = symbolic::__true__();
    EXPECT_TRUE(symbolic::is_true(boolean));

    boolean = symbolic::__false__();
    EXPECT_TRUE(symbolic::is_false(boolean));
}

TEST(SymbolicTest, Pointers) {
    auto ptr = symbolic::__nullptr__();
    EXPECT_TRUE(symbolic::is_pointer(ptr));

    auto intptr = symbolic::symbol("123");
    EXPECT_TRUE(!symbolic::is_pointer(intptr));
}

TEST(SymbolicTest, LogicalExpressions) {
    auto x = symbolic::symbol("x");
    auto y = symbolic::symbol("y");

    auto cond = symbolic::And(symbolic::Eq(x, y), symbolic::Ne(x, symbolic::zero()));
    EXPECT_EQ(cond->__str__(), "And(0 != x, x == y)");

    cond = symbolic::Or(symbolic::Eq(x, y), symbolic::Ne(x, symbolic::zero()));
    EXPECT_EQ(cond->__str__(), "Or(0 != x, x == y)");

    cond = symbolic::Not(symbolic::Eq(x, y));
    EXPECT_EQ(cond->__str__(), "x != y");
}

TEST(SymbolicTest, ArithmeticExpressions) {
    auto x = symbolic::symbol("x");
    auto y = symbolic::symbol("y");

    auto expr = symbolic::add(x, y);
    EXPECT_EQ(expr->__str__(), "x + y");

    expr = symbolic::sub(x, y);
    EXPECT_EQ(expr->__str__(), "x - y");

    expr = symbolic::mul(x, y);
    EXPECT_EQ(expr->__str__(), "x*y");

    expr = symbolic::div(x, y);
    EXPECT_EQ(expr->__str__(), "idiv(x, y)");

    expr = symbolic::min(x, y);
    EXPECT_EQ(expr->__str__(), "min(x, y)");

    expr = symbolic::max(x, y);
    EXPECT_EQ(expr->__str__(), "max(x, y)");

    expr = symbolic::pow(x, y);
    EXPECT_EQ(expr->__str__(), "x**y");
}

TEST(SymbolicTest, eq) {
    auto x = symbolic::symbol("x");
    auto one = symbolic::one();
    auto num = symbolic::integer(1);

    EXPECT_TRUE(symbolic::eq(one, num));
    EXPECT_FALSE(symbolic::eq(one, x));
}

TEST(SymbolicTest, uses) {
    auto x = symbolic::symbol("x");
    auto y = symbolic::symbol("y");
    auto z = symbolic::symbol("z");

    auto expr = symbolic::add(x, y);
    EXPECT_TRUE(symbolic::uses(expr, x));
    EXPECT_TRUE(symbolic::uses(expr, y));
    EXPECT_FALSE(symbolic::uses(expr, z));
}

TEST(SymbolicTest, atoms) {
    auto x = symbolic::symbol("x");
    auto y = symbolic::symbol("y");
    auto z = symbolic::symbol("z");

    auto expr = symbolic::add(x, y);
    auto atoms = symbolic::atoms(expr);
    EXPECT_EQ(atoms.size(), 2);
    EXPECT_TRUE(atoms.find(x) != atoms.end());
    EXPECT_TRUE(atoms.find(y) != atoms.end());
    EXPECT_FALSE(atoms.find(z) != atoms.end());
}

TEST(SymbolicTest, subs) {
    auto x = symbolic::symbol("x");
    auto y = symbolic::symbol("y");
    auto z = symbolic::symbol("z");

    auto expr = symbolic::add(x, y);
    auto new_expr = symbolic::subs(expr, x, z);
    EXPECT_EQ(new_expr->__str__(), "y + z");
}

TEST(SymbolicTest, has) {
    auto x = symbolic::symbol("x");
    auto inf = SymEngine::Inf;

    auto expr = symbolic::max(x, inf);
    EXPECT_TRUE(symbolic::has<SymEngine::Infty>(expr));
}
