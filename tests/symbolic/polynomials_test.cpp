#include "sdfg/symbolic/polynomials.h"

#include <gtest/gtest.h>

using namespace sdfg;

TEST(PolynomialsTest, Linear_1D) {
    auto x = symbolic::symbol("x");
    auto m = symbolic::integer(2);
    auto b = symbolic::integer(1);

    auto expr = symbolic::add(symbolic::mul(m, x), b);

    symbolic::SymbolVec vars = {x};
    auto poly = symbolic::polynomial(expr, vars);

    auto coeffs = symbolic::affine_coefficients(poly, vars);
    EXPECT_EQ(coeffs.size(), 2);
    EXPECT_TRUE(symbolic::eq(coeffs[x], m));
    EXPECT_TRUE(symbolic::eq(coeffs[symbolic::symbol("__daisy_constant__")], b));
}

TEST(PolynomialsTest, Linear_2D) {
    auto x = symbolic::symbol("x");
    auto y = symbolic::symbol("y");
    auto m1 = symbolic::integer(2);
    auto b1 = symbolic::integer(1);
    auto m2 = symbolic::integer(3);
    auto b2 = symbolic::integer(4);

    auto expr = symbolic::add(symbolic::mul(m1, x), b1);
    expr = symbolic::add(expr, symbolic::mul(m2, y));
    expr = symbolic::add(expr, b2);

    symbolic::SymbolVec vars = {x, y};
    auto poly = symbolic::polynomial(expr, vars);

    auto coeffs = symbolic::affine_coefficients(poly, vars);
    EXPECT_EQ(coeffs.size(), 3);
    EXPECT_TRUE(symbolic::eq(coeffs[x], m1));
    EXPECT_TRUE(symbolic::eq(coeffs[y], m2));
    EXPECT_TRUE(
        symbolic::eq(coeffs[symbolic::symbol("__daisy_constant__")], symbolic::add(b1, b2)));
}

TEST(PolynomialsTest, Degree2) {
    auto x = symbolic::symbol("x");

    auto expr = symbolic::mul(x, x);

    symbolic::SymbolVec vars = {x};
    auto poly = symbolic::polynomial(expr, vars);

    auto coeffs = symbolic::affine_coefficients(poly, vars);
    EXPECT_EQ(coeffs.size(), 0);
}
