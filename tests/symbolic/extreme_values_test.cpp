#include "sdfg/symbolic/extreme_values.h"

#include <gtest/gtest.h>

using namespace sdfg;

TEST(ExtremeValuesTest, Symbol_Integral) {
    auto a = symbolic::symbol("a");

    auto lb = symbolic::integer(1);
    auto ub = symbolic::integer(2);

    symbolic::Assumption assum = symbolic::Assumption(a);
    assum.lower_bound(lb);
    assum.upper_bound(ub);

    symbolic::Assumptions assums;
    assums.insert({a, assum});

    auto min = symbolic::minimum(a, assums);
    EXPECT_TRUE(symbolic::eq(min, lb));

    auto max = symbolic::maximum(a, assums);
    EXPECT_TRUE(symbolic::eq(max, ub));
}

TEST(ExtremeValuesTest, Symbol_Symbolic) {
    auto a = symbolic::symbol("a");
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");

    symbolic::Assumption assum_a = symbolic::Assumption(a);
    assum_a.lower_bound(N);
    assum_a.upper_bound(M);

    symbolic::Assumptions assums;
    assums.insert({a, assum_a});

    auto min = symbolic::minimum(a, assums);
    EXPECT_TRUE(symbolic::eq(min, N));

    auto max = symbolic::maximum(a, assums);
    EXPECT_TRUE(symbolic::eq(max, M));
}

TEST(ExtremeValuesTest, Linear_Integral) {
    auto a = symbolic::symbol("a");

    auto lb = symbolic::integer(1);
    auto ub = symbolic::integer(2);

    symbolic::Assumption assum = symbolic::Assumption(a);
    assum.lower_bound(lb);
    assum.upper_bound(ub);

    symbolic::Assumptions assums;
    assums.insert({a, assum});

    auto expr = symbolic::add(symbolic::mul(symbolic::integer(4), a), symbolic::integer(5));

    auto min = symbolic::minimum(expr, assums);
    EXPECT_TRUE(symbolic::eq(min, symbolic::integer(9)));

    auto max = symbolic::maximum(expr, assums);
    EXPECT_TRUE(symbolic::eq(max, symbolic::integer(13)));
}

TEST(ExtremeValuesTest, Linear_Symbolic) {
    auto a = symbolic::symbol("a");
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");

    symbolic::Assumption assum = symbolic::Assumption(a);
    assum.lower_bound(N);
    assum.upper_bound(M);

    symbolic::Assumptions assums;
    assums.insert({a, assum});

    auto expr = symbolic::add(symbolic::mul(symbolic::integer(4), a), symbolic::integer(5));

    auto min = symbolic::minimum(expr, assums);
    auto expr_lb = symbolic::add(symbolic::min(symbolic::mul(symbolic::integer(4), M),
                                               symbolic::mul(symbolic::integer(4), N)),
                                 symbolic::integer(5));
    EXPECT_TRUE(symbolic::eq(min, expr_lb));

    auto max = symbolic::maximum(expr, assums);
    auto expr_ub = symbolic::add(symbolic::max(symbolic::mul(symbolic::integer(4), M),
                                               symbolic::mul(symbolic::integer(4), N)),
                                 symbolic::integer(5));
    EXPECT_TRUE(symbolic::eq(max, expr_ub));
}

TEST(ExtremeValuesTest, Max_Integral) {
    auto a = symbolic::symbol("a");
    auto b = symbolic::symbol("b");

    auto lb_a = symbolic::integer(1);
    auto ub_a = symbolic::integer(2);

    auto lb_b = symbolic::integer(3);
    auto ub_b = symbolic::integer(4);

    symbolic::Assumption assum_a = symbolic::Assumption(a);
    assum_a.lower_bound(lb_a);
    assum_a.upper_bound(ub_a);

    symbolic::Assumption assum_b = symbolic::Assumption(b);
    assum_b.lower_bound(lb_b);
    assum_b.upper_bound(ub_b);

    symbolic::Assumptions assums;
    assums.insert({a, assum_a});
    assums.insert({b, assum_b});

    auto expr = symbolic::max(a, b);

    auto min = symbolic::minimum(expr, assums);
    EXPECT_TRUE(symbolic::eq(min, symbolic::integer(1)));

    auto max = symbolic::maximum(expr, assums);
    EXPECT_TRUE(symbolic::eq(max, symbolic::integer(4)));
}

TEST(ExtremeValuesTest, Max_Symbolic) {
    auto a = symbolic::symbol("a");
    auto b = symbolic::symbol("b");
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    auto N_ = symbolic::symbol("N_");
    auto M_ = symbolic::symbol("M_");

    auto lb_a = symbolic::symbol("N");
    auto ub_a = symbolic::symbol("M");

    auto lb_b = symbolic::symbol("N_");
    auto ub_b = symbolic::symbol("M_");

    symbolic::Assumption assum_a = symbolic::Assumption(a);
    assum_a.lower_bound(lb_a);
    assum_a.upper_bound(ub_a);

    symbolic::Assumption assum_b = symbolic::Assumption(b);
    assum_b.lower_bound(lb_b);
    assum_b.upper_bound(ub_b);

    symbolic::Assumptions assums;
    assums.insert({a, assum_a});
    assums.insert({b, assum_b});

    auto expr = symbolic::max(a, b);

    auto min = symbolic::minimum(expr, assums);
    EXPECT_TRUE(symbolic::eq(min, symbolic::min(N, N_)));

    auto max = symbolic::maximum(expr, assums);
    EXPECT_TRUE(symbolic::eq(max, symbolic::max(M, M_)));
}

TEST(ExtremeValuesTest, Min_Integral) {
    auto a = symbolic::symbol("a");
    auto b = symbolic::symbol("b");

    auto lb_a = symbolic::integer(1);
    auto ub_a = symbolic::integer(2);

    auto lb_b = symbolic::integer(3);
    auto ub_b = symbolic::integer(4);

    symbolic::Assumption assum_a = symbolic::Assumption(a);
    assum_a.lower_bound(lb_a);
    assum_a.upper_bound(ub_a);

    symbolic::Assumption assum_b = symbolic::Assumption(b);
    assum_b.lower_bound(lb_b);
    assum_b.upper_bound(ub_b);

    symbolic::Assumptions assums;
    assums.insert({a, assum_a});
    assums.insert({b, assum_b});

    auto expr = symbolic::min(a, b);

    auto min = symbolic::minimum(expr, assums);
    EXPECT_TRUE(symbolic::eq(min, symbolic::integer(1)));

    auto max = symbolic::maximum(expr, assums);
    EXPECT_TRUE(symbolic::eq(max, symbolic::integer(4)));
}

TEST(ExtremeValuesTest, Min_Symbolic) {
    auto a = symbolic::symbol("a");
    auto b = symbolic::symbol("b");
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    auto N_ = symbolic::symbol("N_");
    auto M_ = symbolic::symbol("M_");

    auto lb_a = symbolic::symbol("N");
    auto ub_a = symbolic::symbol("M");

    auto lb_b = symbolic::symbol("N_");
    auto ub_b = symbolic::symbol("M_");

    symbolic::Assumption assum_a = symbolic::Assumption(a);
    assum_a.lower_bound(lb_a);
    assum_a.upper_bound(ub_a);

    symbolic::Assumption assum_b = symbolic::Assumption(b);
    assum_b.lower_bound(lb_b);
    assum_b.upper_bound(ub_b);

    symbolic::Assumptions assums;
    assums.insert({a, assum_a});
    assums.insert({b, assum_b});

    auto expr = symbolic::min(a, b);

    auto min = symbolic::minimum(expr, assums);
    EXPECT_TRUE(symbolic::eq(min, symbolic::min(N, N_)));

    auto max = symbolic::maximum(expr, assums);
    EXPECT_TRUE(symbolic::eq(max, symbolic::max(M, M_)));
}
