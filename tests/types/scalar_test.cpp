#include "sdfg/types/scalar.h"

#include <gtest/gtest.h>

using namespace sdfg;

TEST(ScalarTest, Init) {
    types::Scalar s(types::PrimitiveType::Int32);
    EXPECT_EQ(s.primitive_type(), types::PrimitiveType::Int32);
}

TEST(ScalarTest, Symbol) {
    types::Scalar s(types::PrimitiveType::UInt32);
    EXPECT_TRUE(s.is_symbol());

    types::Scalar s2(types::PrimitiveType::Float);
    EXPECT_FALSE(s2.is_symbol());
}

TEST(ScalarTest, Equal) {
    types::Scalar s(types::PrimitiveType::UInt32);
    types::Scalar s2(types::PrimitiveType::UInt32);
    types::Scalar s3(types::PrimitiveType::Float);
    EXPECT_EQ(s, s2);
    EXPECT_NE(s, s3);
}

TEST(ScalarTest, TypeId) {
    types::Scalar s(types::PrimitiveType::Float);
    EXPECT_EQ(s.type_id(), types::TypeID::Scalar);
}

TEST(ScalarTest, Clone) {
    types::Scalar s(types::PrimitiveType::Float);
    auto s2 = s.clone();
    EXPECT_EQ(s.primitive_type(), s2->primitive_type());
}

TEST(ScalarTest, AsSigned) {
    types::Scalar s(types::PrimitiveType::UInt32);
    auto s2 = s.as_signed();
    EXPECT_EQ(s2.primitive_type(), types::PrimitiveType::Int32);
}

TEST(ScalarTest, AsUnsigned) {
    types::Scalar s(types::PrimitiveType::Int32);
    auto s2 = s.as_unsigned();
    EXPECT_EQ(s2.primitive_type(), types::PrimitiveType::UInt32);
}
