#include "sdfg/symbolic/analysis.h"

#include <gtest/gtest.h>

#include "sdfg/symbolic/symbolic.h"

using namespace sdfg;

TEST(AnalysisTest, Polynomial) {
    auto x = symbolic::symbol("x");
    auto m = symbolic::symbol("m");
    auto b = symbolic::symbol("b");

    auto constant = b;
    auto constant_poly = symbolic::polynomial(constant, x);
    EXPECT_EQ(constant_poly->get_degree(), 0);
    EXPECT_TRUE(symbolic::eq(constant_poly->get_coeff(0), b));

    auto linear = symbolic::add(symbolic::mul(x, m), b);
    auto linear_poly = symbolic::polynomial(linear, x);
    EXPECT_EQ(linear_poly->get_degree(), 1);
    EXPECT_TRUE(symbolic::eq(linear_poly->get_coeff(0), b));
    EXPECT_TRUE(symbolic::eq(linear_poly->get_coeff(1), m));

    auto quadratic = symbolic::add(symbolic::mul(symbolic::mul(x, x), m), b);
    auto quadratic_poly = symbolic::polynomial(quadratic, x);
    EXPECT_EQ(quadratic_poly->get_degree(), 2);
    EXPECT_TRUE(symbolic::eq(quadratic_poly->get_coeff(0), b));
    EXPECT_TRUE(symbolic::eq(quadratic_poly->get_coeff(1), SymEngine::integer(0)));
    EXPECT_TRUE(symbolic::eq(quadratic_poly->get_coeff(2), m));

    auto cubic = symbolic::add(symbolic::mul(symbolic::mul(symbolic::mul(x, x), x), m), b);
    auto cubic_poly = symbolic::polynomial(cubic, x);
    EXPECT_EQ(cubic_poly->get_degree(), 3);
    EXPECT_TRUE(symbolic::eq(cubic_poly->get_coeff(0), b));
    EXPECT_TRUE(symbolic::eq(cubic_poly->get_coeff(1), SymEngine::integer(0)));
    EXPECT_TRUE(symbolic::eq(cubic_poly->get_coeff(2), SymEngine::integer(0)));
    EXPECT_TRUE(symbolic::eq(cubic_poly->get_coeff(3), m));

    auto log = symbolic::add(symbolic::mul(symbolic::log(x), m), b);
    EXPECT_EQ(symbolic::polynomial(log, x), SymEngine::null);
}

TEST(AnalysisTest, Affine) {
    auto x = symbolic::symbol("x");
    auto m = symbolic::symbol("m");
    auto b = symbolic::symbol("b");

    auto constant = b;
    auto constant_affine = symbolic::affine(constant, x);
    EXPECT_TRUE(symbolic::eq(constant_affine.first, symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(constant_affine.second, b));

    auto linear = symbolic::add(symbolic::mul(x, m), b);
    auto linear_affine = symbolic::affine(linear, x);
    EXPECT_TRUE(symbolic::eq(linear_affine.first, m));
    EXPECT_TRUE(symbolic::eq(linear_affine.second, b));

    auto quadratic = symbolic::add(symbolic::mul(symbolic::mul(x, x), m), b);
    auto quadratic_affine = symbolic::affine(quadratic, x);
    EXPECT_EQ(quadratic_affine.first, SymEngine::null);
    EXPECT_EQ(quadratic_affine.second, SymEngine::null);

    auto m2 = symbolic::symbol("m2");
    auto linear_combination = symbolic::add(symbolic::mul(x, m), symbolic::mul(x, m2));
    auto linear_combination_affine = symbolic::affine(linear_combination, x);
    EXPECT_TRUE(symbolic::eq(linear_combination_affine.first, symbolic::add(m, m2)));
}

TEST(AnalysisTest, Strict_Monotonicity) {
    auto x = symbolic::symbol("x");

    auto affine = symbolic::add(symbolic::mul(x, symbolic::integer(2)), symbolic::integer(1));
    EXPECT_EQ(symbolic::strict_monotonicity_affine(affine, x), symbolic::Sign::POSITIVE);

    auto tan = symbolic::tan(x);
    EXPECT_EQ(symbolic::strict_monotonicity(tan, x), symbolic::Sign::NONE);
}

TEST(AnalysisTest, Strict_Monotonicity_Affine) {
    auto x = symbolic::symbol("x");
    auto m = symbolic::symbol("m");
    auto b = symbolic::symbol("b");

    auto positive = symbolic::add(symbolic::mul(x, symbolic::integer(2)), b);
    EXPECT_EQ(symbolic::strict_monotonicity_affine(positive, x), symbolic::Sign::POSITIVE);

    auto negative = symbolic::add(symbolic::mul(x, symbolic::integer(-2)), b);
    EXPECT_EQ(symbolic::strict_monotonicity_affine(negative, x), symbolic::Sign::NEGATIVE);

    auto constant = b;
    EXPECT_EQ(symbolic::strict_monotonicity_affine(constant, x), symbolic::Sign::NONE);

    auto linear = symbolic::add(symbolic::mul(x, m), b);
    symbolic::Assumptions assumptions;
    assumptions.insert({m, symbolic::Assumption(m)});
    EXPECT_EQ(symbolic::strict_monotonicity_affine(linear, x, assumptions), symbolic::Sign::NONE);
    EXPECT_EQ(symbolic::strict_monotonicity_affine(linear, assumptions), symbolic::Sign::NONE);

    assumptions.at(m).lower_bound(symbolic::integer(1));
    assumptions.at(m).upper_bound(symbolic::infty(1));
    EXPECT_EQ(symbolic::strict_monotonicity_affine(linear, x, assumptions),
              symbolic::Sign::POSITIVE);
    EXPECT_EQ(symbolic::strict_monotonicity_affine(linear, assumptions), symbolic::Sign::NONE);

    assumptions.at(m).lower_bound(symbolic::infty(-1));
    assumptions.at(m).upper_bound(symbolic::integer(-1));
    EXPECT_EQ(symbolic::strict_monotonicity_affine(linear, x, assumptions),
              symbolic::Sign::NEGATIVE);
    EXPECT_EQ(symbolic::strict_monotonicity_affine(linear, assumptions), symbolic::Sign::NONE);

    assumptions.at(m).lower_bound(symbolic::integer(1));
    assumptions.at(m).upper_bound(symbolic::integer(100));
    EXPECT_EQ(symbolic::strict_monotonicity_affine(linear, x, assumptions),
              symbolic::Sign::POSITIVE);
    EXPECT_EQ(symbolic::strict_monotonicity_affine(linear, assumptions), symbolic::Sign::NONE);

    assumptions.at(m).lower_bound(symbolic::integer(-100));
    assumptions.at(m).upper_bound(symbolic::integer(-1));
    EXPECT_EQ(symbolic::strict_monotonicity_affine(linear, x, assumptions),
              symbolic::Sign::NEGATIVE);
    EXPECT_EQ(symbolic::strict_monotonicity_affine(linear, assumptions), symbolic::Sign::NONE);

    assumptions.insert({x, symbolic::Assumption(x)});
    assumptions.at(m).lower_bound(symbolic::integer(1));
    assumptions.at(m).upper_bound(symbolic::infty(1));
    assumptions.at(x).lower_bound(symbolic::integer(1));
    assumptions.at(x).upper_bound(symbolic::infty(1));
    EXPECT_EQ(symbolic::strict_monotonicity_affine(linear, assumptions), symbolic::Sign::POSITIVE);

    assumptions.at(m).lower_bound(symbolic::integer(-1));
    assumptions.at(m).upper_bound(symbolic::infty(-1));
    assumptions.at(x).lower_bound(symbolic::integer(1));
    assumptions.at(x).upper_bound(symbolic::infty(1));
    EXPECT_EQ(symbolic::strict_monotonicity_affine(linear, assumptions), symbolic::Sign::NONE);

    assumptions.at(m).lower_bound(symbolic::integer(-1));
    assumptions.at(m).upper_bound(symbolic::infty(-1));
    assumptions.at(x).lower_bound(symbolic::integer(-1));
    assumptions.at(x).upper_bound(symbolic::infty(-1));
    EXPECT_EQ(symbolic::strict_monotonicity_affine(linear, assumptions), symbolic::Sign::NONE);

    assumptions.at(m).lower_bound(symbolic::integer(1));
    assumptions.at(m).upper_bound(symbolic::integer(100));
    assumptions.at(x).lower_bound(symbolic::integer(1));
    assumptions.at(x).upper_bound(symbolic::integer(100));
    EXPECT_EQ(symbolic::strict_monotonicity_affine(linear, assumptions), symbolic::Sign::POSITIVE);
}

TEST(AnalysisTest, Contiguity) {
    auto x = symbolic::symbol("x");

    auto linear = symbolic::add(symbolic::mul(x, symbolic::integer(1)), symbolic::integer(1));
    EXPECT_TRUE(symbolic::contiguity(linear, x));

    auto tan = symbolic::tan(x);
    EXPECT_FALSE(symbolic::contiguity(tan, x));
}

TEST(AnalysisTest, Contiguity_Affine) {
    auto x = symbolic::symbol("x");
    auto m = symbolic::symbol("m");
    auto b = symbolic::symbol("b");

    auto contiguous = symbolic::add(symbolic::mul(x, symbolic::integer(1)), b);
    EXPECT_TRUE(symbolic::contiguity_affine(contiguous, x));

    auto non_contiguous = symbolic::add(symbolic::mul(x, symbolic::integer(2)), b);
    EXPECT_FALSE(symbolic::contiguity_affine(non_contiguous, x));

    auto linear = symbolic::add(symbolic::mul(x, m), b);
    symbolic::Assumptions assumptions;
    assumptions.insert({m, symbolic::Assumption(m)});
    EXPECT_FALSE(symbolic::contiguity_affine(linear, x, assumptions));

    assumptions.at(m).lower_bound(symbolic::integer(1));
    assumptions.at(m).upper_bound(symbolic::integer(1));
    EXPECT_TRUE(symbolic::contiguity_affine(linear, x, assumptions));
}

TEST(AnalysisTest, atom_extraction) {
    auto x = symbolic::symbol("x");
    auto m = symbolic::symbol("m");
    auto b = symbolic::symbol("b");

    auto constant = b;
    auto constant_atoms = symbolic::atoms(constant);
    EXPECT_EQ(constant_atoms.size(), 1);
    EXPECT_TRUE(symbolic::eq(*constant_atoms.begin(), b));

    auto linear = symbolic::add(symbolic::mul(x, m), b);
    auto linear_atoms = symbolic::atoms(linear);
    EXPECT_EQ(linear_atoms.size(), 3);
    bool found_x = false, found_m = false, found_b = false;
    for (auto atom : linear_atoms) {
        if (symbolic::eq(atom, x)) found_x = true;
        if (symbolic::eq(atom, m)) found_m = true;
        if (symbolic::eq(atom, b)) found_b = true;
    }
    EXPECT_TRUE(found_x);
    EXPECT_TRUE(found_m);
    EXPECT_TRUE(found_b);

    auto quadratic = symbolic::add(symbolic::mul(symbolic::mul(x, x), m), b);
    auto quadratic_atoms = symbolic::atoms(quadratic);
    EXPECT_EQ(quadratic_atoms.size(), 3);
    found_x = false, found_m = false, found_b = false;
    for (auto atom : quadratic_atoms) {
        if (symbolic::eq(atom, x)) found_x = true;
        if (symbolic::eq(atom, m)) found_m = true;
        if (symbolic::eq(atom, b)) found_b = true;
    }
    EXPECT_TRUE(found_x);
    EXPECT_TRUE(found_m);
    EXPECT_TRUE(found_b);

    auto cubic = symbolic::add(symbolic::mul(symbolic::mul(symbolic::mul(x, x), x), m), b);
    auto cubic_atoms = symbolic::atoms(cubic);
    EXPECT_EQ(cubic_atoms.size(), 3);
    found_x = false, found_m = false, found_b = false;
    for (auto atom : cubic_atoms) {
        if (symbolic::eq(atom, x)) found_x = true;
        if (symbolic::eq(atom, m)) found_m = true;
        if (symbolic::eq(atom, b)) found_b = true;
    }
    EXPECT_TRUE(found_x);
    EXPECT_TRUE(found_m);
    EXPECT_TRUE(found_b);

    auto log = symbolic::add(symbolic::mul(symbolic::log(x), m), b);
    auto log_atoms = symbolic::atoms(log);
    EXPECT_EQ(log_atoms.size(), 3);
    found_x = false, found_m = false, found_b = false;
    for (auto atom : log_atoms) {
        if (symbolic::eq(atom, x)) found_x = true;
        if (symbolic::eq(atom, m)) found_m = true;
        if (symbolic::eq(atom, b)) found_b = true;
    }
    EXPECT_TRUE(found_x);
    EXPECT_TRUE(found_m);
    EXPECT_TRUE(found_b);

    auto tan = symbolic::tan(x);
    auto tan_atoms = symbolic::atoms(tan);
    EXPECT_EQ(tan_atoms.size(), 1);
    EXPECT_TRUE(symbolic::eq(*tan_atoms.begin(), x));

    auto sin = symbolic::sin(x);
    auto sin_atoms = symbolic::atoms(sin);
    EXPECT_EQ(sin_atoms.size(), 1);
    EXPECT_TRUE(symbolic::eq(*sin_atoms.begin(), x));

    auto cos = symbolic::cos(x);
    auto cos_atoms = symbolic::atoms(cos);
    EXPECT_EQ(cos_atoms.size(), 1);
    EXPECT_TRUE(symbolic::eq(*cos_atoms.begin(), x));

    auto neg = symbolic::mul(symbolic::integer(-1), x);
    auto neg_atoms = symbolic::atoms(neg);
    EXPECT_EQ(neg_atoms.size(), 1);
    EXPECT_TRUE(symbolic::eq(*neg_atoms.begin(), x));
}

TEST(AnalysisTest, infinity_analysis) {
    auto x = symbolic::infty(1);
    auto m = symbolic::symbol("m");
    auto b = symbolic::symbol("b");

    auto mult = symbolic::mul(x, m);
    EXPECT_FALSE(symbolic::eq(mult, x));

    auto add = symbolic::add(x, b);
    EXPECT_FALSE(symbolic::eq(add, x));

    auto sub = symbolic::sub(x, b);
    EXPECT_FALSE(symbolic::eq(sub, x));

    auto div = symbolic::div(x, m);
    EXPECT_FALSE(symbolic::eq(div, x));

    auto mult_pos = symbolic::mul(x, symbolic::integer(42));
    EXPECT_TRUE(symbolic::eq(mult_pos, x));

    auto mult_neg = symbolic::mul(x, symbolic::integer(-42));
    EXPECT_TRUE(symbolic::eq(mult_neg, symbolic::infty(-1)));

    auto add_pos = symbolic::add(x, symbolic::integer(42));
    EXPECT_TRUE(symbolic::eq(add_pos, x));

    auto add_neg = symbolic::add(x, symbolic::integer(-42));
    EXPECT_TRUE(symbolic::eq(add_neg, x));

    auto sub_pos = symbolic::sub(x, symbolic::integer(42));
    EXPECT_TRUE(symbolic::eq(sub_pos, x));

    auto sub_neg = symbolic::sub(x, symbolic::integer(-42));
    EXPECT_TRUE(symbolic::eq(sub_neg, x));

    auto div_pos = symbolic::div(x, symbolic::integer(42));
    EXPECT_TRUE(symbolic::eq(div_pos, x));

    auto div_neg = symbolic::div(x, symbolic::integer(-42));
    EXPECT_TRUE(symbolic::eq(div_neg, symbolic::infty(-1)));
}

TEST(AnalysisTest, contains_pos_infinity) {
    auto x = symbolic::infty(1);
    auto m = symbolic::symbol("m");
    auto b = symbolic::symbol("b");

    auto constant = b;
    EXPECT_FALSE(symbolic::contains_infinity(constant));
    EXPECT_TRUE(symbolic::contains_infinity(x));

    auto linear = symbolic::add(symbolic::mul(x, m), b);
    EXPECT_TRUE(symbolic::contains_infinity(linear));

    auto quadratic = symbolic::add(symbolic::mul(symbolic::mul(x, x), m), b);
    EXPECT_TRUE(symbolic::contains_infinity(quadratic));

    auto cubic = symbolic::add(symbolic::mul(symbolic::mul(symbolic::mul(x, x), x), m), b);
    EXPECT_TRUE(symbolic::contains_infinity(cubic));

    auto log = symbolic::add(symbolic::mul(symbolic::log(x), m), b);
    EXPECT_TRUE(symbolic::contains_infinity(log));

    auto neg = symbolic::mul(symbolic::integer(-1), x);
    EXPECT_TRUE(symbolic::contains_infinity(neg));

    auto mult = symbolic::mul(x, m);
    EXPECT_TRUE(symbolic::contains_infinity(mult));

    auto add = symbolic::add(x, b);
    EXPECT_TRUE(symbolic::contains_infinity(add));

    auto sub = symbolic::sub(x, b);
    EXPECT_TRUE(symbolic::contains_infinity(sub));

    auto div = symbolic::div(x, m);
    EXPECT_TRUE(symbolic::contains_infinity(div));
}

TEST(AnalysisTest, contains_neg_infinity) {
    auto x = symbolic::infty(-1);
    auto m = symbolic::symbol("m");
    auto b = symbolic::symbol("b");

    auto constant = b;
    EXPECT_FALSE(symbolic::contains_infinity(constant));
    EXPECT_TRUE(symbolic::contains_infinity(x));

    auto linear = symbolic::add(symbolic::mul(x, m), b);
    EXPECT_TRUE(symbolic::contains_infinity(linear));

    auto quadratic = symbolic::add(symbolic::mul(symbolic::mul(x, x), m), b);
    EXPECT_TRUE(symbolic::contains_infinity(quadratic));

    auto cubic = symbolic::add(symbolic::mul(symbolic::mul(symbolic::mul(x, x), x), m), b);
    EXPECT_TRUE(symbolic::contains_infinity(cubic));

    auto log = symbolic::add(symbolic::mul(symbolic::log(x), m), b);
    EXPECT_TRUE(symbolic::contains_infinity(log));

    auto neg = symbolic::mul(symbolic::integer(-1), x);
    EXPECT_TRUE(symbolic::contains_infinity(neg));

    auto mult = symbolic::mul(x, m);
    EXPECT_TRUE(symbolic::contains_infinity(mult));

    auto add = symbolic::add(x, b);
    EXPECT_TRUE(symbolic::contains_infinity(add));

    auto sub = symbolic::sub(x, b);
    EXPECT_TRUE(symbolic::contains_infinity(sub));

    auto div = symbolic::div(x, m);
    EXPECT_TRUE(symbolic::contains_infinity(div));
}

TEST(AnalysisTest, upper_bound_analysis) {
    auto x = symbolic::symbol("x");
    auto m = symbolic::symbol("m");
    auto b = symbolic::symbol("b");

    symbolic::Assumptions assumptions_empty;
    symbolic::Assumptions assumptions;
    assumptions.insert({m, symbolic::Assumption(m)});
    assumptions.insert({b, symbolic::Assumption(b)});
    assumptions.insert({x, symbolic::Assumption(x)});
    assumptions[m].lower_bound(symbolic::integer(1));
    assumptions[m].upper_bound(symbolic::integer(32));
    assumptions[b].lower_bound(symbolic::integer(2));
    assumptions[b].upper_bound(symbolic::integer(32));

    EXPECT_TRUE(symbolic::eq(symbolic::upper_bound_analysis(symbolic::integer(5), assumptions),
                             symbolic::integer(5)));

    auto constant = b;
    EXPECT_TRUE(
        symbolic::eq(symbolic::upper_bound_analysis(constant, assumptions), symbolic::integer(32)));

    auto linear = symbolic::add(symbolic::mul(x, m), b);
    EXPECT_TRUE(
        symbolic::eq(symbolic::upper_bound_analysis(linear, assumptions), symbolic::infty(1)));
    EXPECT_TRUE(symbolic::eq(symbolic::upper_bound_analysis(linear, assumptions_empty),
                             symbolic::infty(1)));

    assumptions[m].lower_bound(symbolic::integer(1));
    assumptions[m].upper_bound(symbolic::integer(3));
    assumptions[b].lower_bound(symbolic::integer(5));
    assumptions[b].upper_bound(symbolic::integer(10));
    assumptions[x].lower_bound(symbolic::integer(2));
    assumptions[x].upper_bound(symbolic::integer(32));

    auto linear2 = symbolic::add(symbolic::mul(x, m), b);
    EXPECT_TRUE(
        symbolic::eq(symbolic::upper_bound_analysis(linear2, assumptions),
                     symbolic::add(symbolic::mul(symbolic::integer(3), symbolic::integer(32)),
                                   symbolic::integer(10))));
}

TEST(AnalysisTest, lower_bound_analysis) {
    auto x = symbolic::symbol("x");
    auto m = symbolic::symbol("m");
    auto b = symbolic::symbol("b");

    symbolic::Assumptions assumptions_empty;
    symbolic::Assumptions assumptions;
    assumptions.insert({m, symbolic::Assumption(m)});
    assumptions.insert({b, symbolic::Assumption(b)});
    assumptions.insert({x, symbolic::Assumption(x)});
    assumptions[m].lower_bound(symbolic::integer(1));
    assumptions[m].upper_bound(symbolic::integer(32));
    assumptions[b].lower_bound(symbolic::integer(2));
    assumptions[b].upper_bound(symbolic::integer(32));

    EXPECT_TRUE(symbolic::eq(symbolic::lower_bound_analysis(symbolic::integer(5), assumptions),
                             symbolic::integer(5)));

    auto constant = b;
    EXPECT_TRUE(
        symbolic::eq(symbolic::lower_bound_analysis(constant, assumptions), symbolic::integer(2)));

    auto linear = symbolic::add(symbolic::mul(x, m), b);
    EXPECT_TRUE(
        symbolic::eq(symbolic::lower_bound_analysis(linear, assumptions), symbolic::infty(-1)));
    EXPECT_TRUE(symbolic::eq(symbolic::lower_bound_analysis(linear, assumptions_empty),
                             symbolic::infty(-1)));

    assumptions[m].lower_bound(symbolic::integer(3));
    assumptions[m].upper_bound(symbolic::integer(32));
    assumptions[b].lower_bound(symbolic::integer(5));
    assumptions[b].upper_bound(symbolic::integer(32));
    assumptions[x].lower_bound(symbolic::integer(2));
    assumptions[x].upper_bound(symbolic::integer(32));

    auto linear2 = symbolic::add(symbolic::mul(x, m), b);
    EXPECT_TRUE(
        symbolic::eq(symbolic::lower_bound_analysis(linear2, assumptions),
                     symbolic::add(symbolic::mul(symbolic::integer(3), symbolic::integer(2)),
                                   symbolic::integer(5))));
}

TEST(AnalysisTest, affine_coefficients_affine) {
    auto x = symbolic::symbol("x");
    auto y = symbolic::symbol("y");
    auto m = symbolic::integer(2);
    auto n = symbolic::integer(3);

    symbolic::SymbolicVector vars = {x, y};

    auto expr = symbolic::add(symbolic::add(symbolic::mul(m, x), symbolic::mul(n, y)),
                              symbolic::integer(1));
    auto poly = symbolic::multi_polynomial(expr, vars);

    auto coeffs = symbolic::affine_coefficients(poly, vars);
    EXPECT_EQ(coeffs.size(), 3);
    EXPECT_EQ(coeffs[x], 2);
    EXPECT_EQ(coeffs[y], 3);
    EXPECT_EQ(coeffs[symbolic::symbol("__daisy_constant__")], 1);
}

TEST(AnalysisTest, affine_coefficients_non_affine) {
    auto x = symbolic::symbol("x");
    auto y = symbolic::symbol("y");
    auto m = symbolic::symbol("m");
    auto n = symbolic::symbol("n");

    symbolic::SymbolicVector vars = {x, y};

    auto expr = symbolic::add(symbolic::add(symbolic::mul(m, x), symbolic::mul(n, y)),
                              symbolic::integer(1));
    auto poly = symbolic::multi_polynomial(expr, vars);

    auto coeffs = symbolic::affine_coefficients(poly, vars);
    EXPECT_EQ(coeffs.size(), 0);
}

TEST(AnalysisTest, conjunctive_normal_form) {
    auto a = symbolic::symbol("a");
    auto b = symbolic::symbol("b");

    auto atom_a = symbolic::Eq(a, symbolic::integer(1));
    auto atom_b = symbolic::Le(b, symbolic::integer(2));

    auto expr = symbolic::Or(symbolic::And(atom_a, atom_b),
                             symbolic::And(symbolic::Not(atom_a), symbolic::Not(atom_b)));

    auto cnf = symbolic::conjunctive_normal_form(expr);
    EXPECT_EQ(cnf.size(), 4);
}
