#include "sdfg/symbolic/sets.h"

#include <gtest/gtest.h>

#include "sdfg/symbolic/utils.h"
#include "sdfg/types/scalar.h"

using namespace sdfg;

TEST(DelinearizeTest, delinearize_2d) {
    auto x = symbolic::symbol("x");
    auto y = symbolic::symbol("y");
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    types::Scalar desc(types::PrimitiveType::UInt8);

    auto assum_x = symbolic::Assumption::create(x, desc);
    assum_x.lower_bound(symbolic::zero());
    assum_x.upper_bound(symbolic::sub(N, symbolic::integer(1)));

    auto assum_y = symbolic::Assumption::create(y, desc);
    assum_y.lower_bound(symbolic::zero());
    assum_y.upper_bound(symbolic::sub(M, symbolic::integer(1)));

    auto assum_N = symbolic::Assumption::create(N, desc);
    assum_N.lower_bound(symbolic::integer(1));
    assum_N.upper_bound(symbolic::integer(10));
    assum_N.constant(true);

    auto assum_M = symbolic::Assumption::create(M, desc);
    assum_M.lower_bound(symbolic::integer(1));
    assum_M.upper_bound(symbolic::integer(20));
    assum_M.constant(true);

    symbolic::Assumptions assums;
    assums.insert({x, assum_x});
    assums.insert({y, assum_y});
    assums.insert({N, assum_N});
    assums.insert({M, assum_M});

    auto expr = symbolic::add(symbolic::mul(x, M), y);

    auto expr_delinearized = symbolic::delinearize({expr}, assums);
    EXPECT_EQ(expr_delinearized.size(), 2);
    EXPECT_TRUE(symbolic::eq(expr_delinearized.at(0), x));
    EXPECT_TRUE(symbolic::eq(expr_delinearized.at(1), y));
}

TEST(DelinearizeTest, delinearize_2d_stride_may_be_zero) {
    auto x = symbolic::symbol("x");
    auto y = symbolic::symbol("y");
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    types::Scalar desc(types::PrimitiveType::UInt8);

    auto assum_x = symbolic::Assumption::create(x, desc);
    assum_x.lower_bound(symbolic::zero());
    assum_x.upper_bound(symbolic::sub(N, symbolic::integer(1)));

    auto assum_y = symbolic::Assumption::create(y, desc);
    assum_y.lower_bound(symbolic::zero());
    assum_y.upper_bound(symbolic::sub(M, symbolic::integer(1)));

    auto assum_N = symbolic::Assumption::create(N, desc);
    assum_N.lower_bound(symbolic::integer(0));
    assum_N.upper_bound(symbolic::integer(10));
    assum_N.constant(true);

    auto assum_M = symbolic::Assumption::create(M, desc);
    assum_M.lower_bound(symbolic::integer(0));
    assum_M.upper_bound(symbolic::integer(20));
    assum_M.constant(true);

    symbolic::Assumptions assums;
    assums.insert({x, assum_x});
    assums.insert({y, assum_y});
    assums.insert({N, assum_N});
    assums.insert({M, assum_M});

    auto expr = symbolic::add(symbolic::mul(x, M), y);

    auto expr_delinearized = symbolic::delinearize({expr}, assums);
    EXPECT_EQ(expr_delinearized.size(), 1);
    EXPECT_TRUE(symbolic::eq(expr_delinearized.at(0), expr));
}

TEST(DelinearizeTest, delinearize_2d_symbols_may_be_negative) {
    auto x = symbolic::symbol("x");
    auto y = symbolic::symbol("y");
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    types::Scalar desc(types::PrimitiveType::UInt8);

    auto assum_x = symbolic::Assumption::create(x, desc);
    assum_x.lower_bound(symbolic::integer(-1));
    assum_x.upper_bound(symbolic::sub(N, symbolic::integer(1)));

    auto assum_y = symbolic::Assumption::create(y, desc);
    assum_y.lower_bound(symbolic::integer(-1));
    assum_y.upper_bound(symbolic::sub(M, symbolic::integer(1)));

    auto assum_N = symbolic::Assumption::create(N, desc);
    assum_N.lower_bound(symbolic::integer(1));
    assum_N.upper_bound(symbolic::integer(10));
    assum_N.constant(true);

    auto assum_M = symbolic::Assumption::create(M, desc);
    assum_M.lower_bound(symbolic::integer(1));
    assum_M.upper_bound(symbolic::integer(20));
    assum_M.constant(true);

    symbolic::Assumptions assums;
    assums.insert({x, assum_x});
    assums.insert({y, assum_y});
    assums.insert({N, assum_N});
    assums.insert({M, assum_M});

    auto expr = symbolic::add(symbolic::mul(x, M), y);

    auto expr_delinearized = symbolic::delinearize({expr}, assums);
    EXPECT_EQ(expr_delinearized.size(), 1);
    EXPECT_TRUE(symbolic::eq(expr_delinearized.at(0), expr));
}

TEST(DelinearizeTest, delinearize_3d) {
    auto x = symbolic::symbol("x");
    auto y = symbolic::symbol("y");
    auto z = symbolic::symbol("z");
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    auto K = symbolic::symbol("K");
    types::Scalar desc(types::PrimitiveType::UInt8);

    auto assum_x = symbolic::Assumption::create(x, desc);
    assum_x.lower_bound(symbolic::zero());
    assum_x.upper_bound(symbolic::sub(N, symbolic::integer(1)));

    auto assum_y = symbolic::Assumption::create(y, desc);
    assum_y.lower_bound(symbolic::zero());
    assum_y.upper_bound(symbolic::sub(M, symbolic::integer(1)));

    auto assum_z = symbolic::Assumption::create(z, desc);
    assum_z.lower_bound(symbolic::zero());
    assum_z.upper_bound(symbolic::sub(K, symbolic::integer(1)));

    auto assum_N = symbolic::Assumption::create(N, desc);
    assum_N.lower_bound(symbolic::integer(1));
    assum_N.upper_bound(symbolic::integer(10));
    assum_N.constant(true);

    auto assum_M = symbolic::Assumption::create(M, desc);
    assum_M.lower_bound(symbolic::integer(1));
    assum_M.upper_bound(symbolic::integer(20));
    assum_M.constant(true);

    auto assum_K = symbolic::Assumption::create(K, desc);
    assum_K.lower_bound(symbolic::integer(1));
    assum_K.upper_bound(symbolic::integer(30));
    assum_K.constant(true);

    symbolic::Assumptions assums;
    assums.insert({x, assum_x});
    assums.insert({y, assum_y});
    assums.insert({z, assum_z});
    assums.insert({N, assum_N});
    assums.insert({M, assum_M});
    assums.insert({K, assum_K});

    auto expr =
        symbolic::add(symbolic::add(symbolic::mul(x, symbolic::mul(M, K)), symbolic::mul(y, K)), z);

    auto expr_delinearized = symbolic::delinearize({expr}, assums);
    EXPECT_EQ(expr_delinearized.size(), 3);
    EXPECT_TRUE(symbolic::eq(expr_delinearized.at(0), x));
    EXPECT_TRUE(symbolic::eq(expr_delinearized.at(1), y));
    EXPECT_TRUE(symbolic::eq(expr_delinearized.at(2), z));
}

TEST(SetsTest, is_subset_1d_equivalent) {
    auto x = symbolic::symbol("x");
    types::Scalar desc(types::PrimitiveType::UInt8);

    auto assum_x = symbolic::Assumption::create(x, desc);
    symbolic::Assumptions assums;
    assums.insert({x, assum_x});

    auto expr1 = symbolic::add(x, symbolic::integer(1));
    auto expr2 = symbolic::add(x, symbolic::integer(1));

    EXPECT_TRUE(symbolic::is_subset({expr1}, {expr2}, assums, assums));
}

TEST(SetsTest, is_subset_1d_not_equivalent) {
    auto x = symbolic::symbol("x");
    types::Scalar desc(types::PrimitiveType::UInt8);

    auto assum_x = symbolic::Assumption::create(x, desc);
    symbolic::Assumptions assums;
    assums.insert({x, assum_x});

    auto expr1 = symbolic::add(x, symbolic::integer(1));
    auto expr2 = symbolic::add(x, symbolic::integer(2));

    EXPECT_FALSE(symbolic::is_subset({expr1}, {expr2}, assums, assums));
}

TEST(SetsTest, is_subset_1d_equivalent_rename) {
    auto x = symbolic::symbol("x");
    auto y = symbolic::symbol("y");
    types::Scalar desc(types::PrimitiveType::UInt8);

    auto assum_x = symbolic::Assumption::create(x, desc);
    assum_x.lower_bound(y);
    assum_x.upper_bound(y);

    auto assum_y = symbolic::Assumption::create(y, desc);
    assum_y.lower_bound(x);
    assum_y.upper_bound(x);

    symbolic::Assumptions assums;
    assums.insert({x, assum_x});
    assums.insert({y, assum_y});

    auto expr1 = symbolic::add(x, symbolic::integer(1));
    auto expr2 = symbolic::add(y, symbolic::integer(1));

    EXPECT_TRUE(symbolic::is_subset({expr1}, {expr2}, assums, assums));
}

TEST(SetsTest, is_subset_1d_equivalent_recursive_assumptions) {
    auto x = symbolic::symbol("x");
    auto y = symbolic::symbol("y");
    auto z = symbolic::symbol("z");
    types::Scalar desc(types::PrimitiveType::UInt8);

    auto assum_x = symbolic::Assumption::create(x, desc);
    assum_x.lower_bound(x);
    assum_x.upper_bound(z);

    auto assum_y = symbolic::Assumption::create(y, desc);
    assum_y.lower_bound(z);
    assum_y.upper_bound(y);

    auto assum_z = symbolic::Assumption::create(z, desc);
    assum_z.lower_bound(y);
    assum_z.upper_bound(x);

    symbolic::Assumptions assums;
    assums.insert({x, assum_x});
    assums.insert({y, assum_y});
    assums.insert({z, assum_z});

    auto expr1 = symbolic::add(x, symbolic::integer(1));
    auto expr2 = symbolic::add(y, symbolic::integer(1));

    EXPECT_TRUE(symbolic::is_subset({expr1}, {expr2}, assums, assums));
}

TEST(SetsTest, is_subset_1d_minmax_assumptions) {
    auto x = symbolic::symbol("x");
    auto y = symbolic::symbol("y");
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    auto K = symbolic::symbol("K");
    types::Scalar desc(types::PrimitiveType::UInt8);

    auto assum_x = symbolic::Assumption::create(x, desc);
    assum_x.lower_bound(symbolic::max(N, M));
    assum_x.upper_bound(symbolic::min(M, K));

    auto assum_y = symbolic::Assumption::create(y, desc);
    assum_y.lower_bound(symbolic::max(N, M));
    assum_y.upper_bound(symbolic::min(M, K));

    symbolic::Assumptions assums;
    assums.insert({x, assum_x});
    assums.insert({y, assum_y});

    auto expr1 = symbolic::add(x, symbolic::integer(1));
    auto expr2 = symbolic::add(y, symbolic::integer(1));

    EXPECT_TRUE(symbolic::is_subset({expr1}, {expr2}, assums, assums));
}

TEST(SetsTest, is_disjoint_1d_disjoint) {
    auto x = symbolic::symbol("x");
    types::Scalar desc(types::PrimitiveType::UInt8);

    auto assum_x = symbolic::Assumption::create(x, desc);
    symbolic::Assumptions assums;
    assums.insert({x, assum_x});

    auto expr1 = symbolic::add(x, symbolic::integer(1));
    auto expr2 = symbolic::add(x, symbolic::integer(1));

    EXPECT_FALSE(symbolic::is_disjoint({expr1}, {expr2}, assums, assums));
}

TEST(SetsTest, is_disjoint_1d_not_disjoint) {
    auto x = symbolic::symbol("x");
    types::Scalar desc(types::PrimitiveType::UInt8);

    auto assum_x = symbolic::Assumption::create(x, desc);
    symbolic::Assumptions assums;
    assums.insert({x, assum_x});

    auto expr1 = symbolic::add(x, symbolic::integer(1));
    auto expr2 = symbolic::add(x, symbolic::integer(2));

    EXPECT_FALSE(symbolic::is_disjoint({expr1}, {expr2}, assums, assums));
}
