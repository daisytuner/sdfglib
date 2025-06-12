#include "sdfg/symbolic/sets.h"

#include <gtest/gtest.h>

#include "sdfg/types/scalar.h"

using namespace sdfg;

TEST(SetsTest, is_equivalent_1d_identity) {
    auto x = symbolic::symbol("x");
    types::Scalar desc(types::PrimitiveType::UInt8);

    auto assum_x = symbolic::Assumption::create(x, desc);
    symbolic::Assumptions assums;
    assums.insert({x, assum_x});

    auto expr1 = symbolic::add(x, symbolic::integer(1));
    auto expr2 = symbolic::add(x, symbolic::integer(1));

    EXPECT_TRUE(symbolic::is_equivalent({expr1}, {expr2}, {}, assums));
}

TEST(SetsTest, is_equivalent_1d_disjoint) {
    auto x = symbolic::symbol("x");
    types::Scalar desc(types::PrimitiveType::UInt8);

    auto assum_x = symbolic::Assumption::create(x, desc);
    symbolic::Assumptions assums;
    assums.insert({x, assum_x});

    auto expr1 = symbolic::add(x, symbolic::integer(1));
    auto expr2 = symbolic::add(x, symbolic::integer(2));

    EXPECT_FALSE(symbolic::is_equivalent({expr1}, {expr2}, {}, assums));
}

TEST(SetsTest, is_equivalent_1d_rename) {
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

    EXPECT_TRUE(symbolic::is_equivalent({expr1}, {expr2}, {}, assums));
}

TEST(SetsTest, is_equivalent_1d_recursive_assumptions) {
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

    EXPECT_TRUE(symbolic::is_equivalent({expr1}, {expr2}, {}, assums));
}
