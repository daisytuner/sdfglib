#include "sdfg/symbolic/extreme_values.h"

#include <gtest/gtest.h>

using namespace sdfg;

TEST(ExtremeValuesTest, Symbol) {
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

TEST(ExtremeValuesTest, Linear) {
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

TEST(ExtremeValuesTest, Max) {
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

    auto expr =
        symbolic::max(symbolic::add(symbolic::mul(symbolic::integer(4), a), symbolic::integer(5)),
                      symbolic::add(symbolic::mul(symbolic::integer(6), b), symbolic::integer(7)));

    auto min = symbolic::minimum(expr, assums);
    EXPECT_TRUE(symbolic::eq(min, symbolic::integer(9)));

    auto max = symbolic::maximum(expr, assums);
    EXPECT_TRUE(symbolic::eq(max, symbolic::integer(31)));
}

TEST(ExtremeValuesTest, Min) {
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

    auto expr =
        symbolic::min(symbolic::add(symbolic::mul(symbolic::integer(4), a), symbolic::integer(5)),
                      symbolic::add(symbolic::mul(symbolic::integer(6), b), symbolic::integer(7)));

    auto min = symbolic::minimum(expr, assums);
    EXPECT_TRUE(symbolic::eq(min, symbolic::integer(9)));

    auto max = symbolic::maximum(expr, assums);
    EXPECT_TRUE(symbolic::eq(max, symbolic::integer(31)));
}
