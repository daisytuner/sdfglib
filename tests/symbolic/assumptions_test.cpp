#include "sdfg/symbolic/assumptions.h"

#include <gtest/gtest.h>

using namespace sdfg;

TEST(AssumptionsTest, Init) {
    auto x = symbolic::symbol("x");

    symbolic::Assumption a(x);
    EXPECT_TRUE(symbolic::eq(a.symbol(), x));
    EXPECT_TRUE(symbolic::eq(a.lower_bound(), symbolic::infty(-1)));
    EXPECT_TRUE(symbolic::eq(a.upper_bound(), symbolic::infty(1)));
}

TEST(AssumptionsTest, Domain) {
    auto x = symbolic::symbol("x");

    symbolic::Assumption a(x);
    EXPECT_FALSE(a.is_negative());
    EXPECT_FALSE(a.is_positive());
    EXPECT_FALSE(a.is_nonnegative());
    EXPECT_FALSE(a.is_nonpositive());

    a.lower_bound(symbolic::integer(0));
    EXPECT_FALSE(a.is_negative());
    EXPECT_FALSE(a.is_positive());
    EXPECT_TRUE(a.is_nonnegative());
    EXPECT_FALSE(a.is_nonpositive());

    a.lower_bound(symbolic::integer(1));
    EXPECT_FALSE(a.is_negative());
    EXPECT_TRUE(a.is_positive());
    EXPECT_TRUE(a.is_nonnegative());
    EXPECT_FALSE(a.is_nonpositive());

    a.upper_bound(symbolic::integer(1));
    EXPECT_FALSE(a.is_negative());
    EXPECT_TRUE(a.is_positive());
    EXPECT_TRUE(a.is_nonnegative());
    EXPECT_FALSE(a.is_nonpositive());

    a.lower_bound(symbolic::integer(-1));
    a.upper_bound(symbolic::integer(-1));
    EXPECT_TRUE(a.is_negative());
    EXPECT_FALSE(a.is_positive());
    EXPECT_FALSE(a.is_nonnegative());
    EXPECT_TRUE(a.is_nonpositive());
}

TEST(AssumptionsTest, AssumptionsSet) {
    auto x = symbolic::symbol("x");
    auto y = symbolic::symbol("y");

    symbolic::Assumptions as;
    as[x] = symbolic::Assumption(x);
    as[y] = symbolic::Assumption(y);

    EXPECT_TRUE(symbolic::eq(as[x].symbol(), x));
    EXPECT_TRUE(symbolic::eq(as[y].symbol(), y));
}

TEST(AssumptionsTest, Create) {
    auto x = symbolic::symbol("x");
    auto y = symbolic::symbol("y");

    auto a = symbolic::Assumption::create(x, types::Scalar(types::PrimitiveType::Bool));
    EXPECT_TRUE(symbolic::eq(a.lower_bound(), symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(a.upper_bound(), symbolic::one()));

    a = symbolic::Assumption::create(x, types::Scalar(types::PrimitiveType::UInt8));
    EXPECT_TRUE(symbolic::eq(a.lower_bound(), symbolic::zero()));
    EXPECT_TRUE(
        symbolic::eq(a.upper_bound(), symbolic::integer(std::numeric_limits<uint8_t>::max())));

    a = symbolic::Assumption::create(x, types::Scalar(types::PrimitiveType::Int8));
    EXPECT_TRUE(
        symbolic::eq(a.lower_bound(), symbolic::integer(std::numeric_limits<int8_t>::min())));
    EXPECT_TRUE(
        symbolic::eq(a.upper_bound(), symbolic::integer(std::numeric_limits<int8_t>::max())));

    a = symbolic::Assumption::create(x, types::Scalar(types::PrimitiveType::UInt16));
    EXPECT_TRUE(symbolic::eq(a.lower_bound(), symbolic::zero()));
    EXPECT_TRUE(
        symbolic::eq(a.upper_bound(), symbolic::integer(std::numeric_limits<uint16_t>::max())));

    a = symbolic::Assumption::create(x, types::Scalar(types::PrimitiveType::Int16));
    EXPECT_TRUE(
        symbolic::eq(a.lower_bound(), symbolic::integer(std::numeric_limits<int16_t>::min())));
    EXPECT_TRUE(
        symbolic::eq(a.upper_bound(), symbolic::integer(std::numeric_limits<int16_t>::max())));

    a = symbolic::Assumption::create(x, types::Scalar(types::PrimitiveType::UInt32));
    EXPECT_TRUE(
        symbolic::eq(a.lower_bound(), symbolic::integer(std::numeric_limits<uint32_t>::min())));
    EXPECT_TRUE(
        symbolic::eq(a.upper_bound(), symbolic::integer(std::numeric_limits<uint32_t>::max())));

    a = symbolic::Assumption::create(x, types::Scalar(types::PrimitiveType::Int32));
    EXPECT_TRUE(
        symbolic::eq(a.lower_bound(), symbolic::integer(std::numeric_limits<int32_t>::min())));
    EXPECT_TRUE(
        symbolic::eq(a.upper_bound(), symbolic::integer(std::numeric_limits<int32_t>::max())));

    a = symbolic::Assumption::create(x, types::Scalar(types::PrimitiveType::UInt64));
    EXPECT_TRUE(
        symbolic::eq(a.lower_bound(), symbolic::integer(std::numeric_limits<uint64_t>::min())));
    EXPECT_TRUE(symbolic::eq(a.upper_bound(), symbolic::infty(1)));

    a = symbolic::Assumption::create(x, types::Scalar(types::PrimitiveType::Int64));
    EXPECT_TRUE(symbolic::eq(a.lower_bound(), symbolic::infty(-1)));
    EXPECT_TRUE(symbolic::eq(a.upper_bound(), symbolic::infty(1)));
}

TEST(AssumptionsTest, upper_bounds) {
    auto x = symbolic::symbol("x");
    auto y = symbolic::symbol("y");
    auto z = symbolic::symbol("z");

    symbolic::Assumptions as;

    symbolic::Assumption a1(x);
    a1.upper_bound(y);

    symbolic::Assumption a2(y);
    a2.upper_bound(symbolic::min(symbolic::integer(10), z));

    as[x] = a1;
    as[y] = a2;

    symbolic::SymbolicSet ubs;
    symbolic::upper_bounds(x, as, ubs);
    EXPECT_EQ(ubs.size(), 3);
    EXPECT_TRUE(ubs.find(symbolic::integer(10)) != ubs.end());
    EXPECT_TRUE(ubs.find(z) != ubs.end());
    EXPECT_TRUE(ubs.find(y) != ubs.end());
}

TEST(AssumptionsTest, lower_bounds) {
    auto x = symbolic::symbol("x");
    auto y = symbolic::symbol("y");
    auto z = symbolic::symbol("z");

    symbolic::Assumptions as;

    symbolic::Assumption a1(x);
    a1.lower_bound(y);

    symbolic::Assumption a2(y);
    a2.lower_bound(symbolic::max(symbolic::integer(10), z));

    as[x] = a1;
    as[y] = a2;

    symbolic::SymbolicSet lbs;
    symbolic::lower_bounds(x, as, lbs);
    EXPECT_EQ(lbs.size(), 3);
    EXPECT_TRUE(lbs.find(symbolic::integer(10)) != lbs.end());
    EXPECT_TRUE(lbs.find(z) != lbs.end());
    EXPECT_TRUE(lbs.find(y) != lbs.end());
}
