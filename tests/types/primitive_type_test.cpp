#include <gtest/gtest.h>

#include "sdfg/types/type.h"

using namespace sdfg;

TEST(PrimitiveTypeTest, Enum) {
    EXPECT_EQ(types::PrimitiveType::Void, 0);
    EXPECT_EQ(types::PrimitiveType::Bool, 1);
    EXPECT_EQ(types::PrimitiveType::Int8, 2);
    EXPECT_EQ(types::PrimitiveType::Int16, 3);
    EXPECT_EQ(types::PrimitiveType::Int32, 4);
    EXPECT_EQ(types::PrimitiveType::Int64, 5);
    EXPECT_EQ(types::PrimitiveType::UInt8, 6);
    EXPECT_EQ(types::PrimitiveType::UInt16, 7);
    EXPECT_EQ(types::PrimitiveType::UInt32, 8);
    EXPECT_EQ(types::PrimitiveType::UInt64, 9);
    EXPECT_EQ(types::PrimitiveType::Float, 10);
    EXPECT_EQ(types::PrimitiveType::Double, 11);
}

TEST(PrimitiveTypeTest, ToString) {
    EXPECT_EQ(types::primitive_type_to_string(types::PrimitiveType::Void), "Void");
    EXPECT_EQ(types::primitive_type_to_string(types::PrimitiveType::Bool), "Bool");
    EXPECT_EQ(types::primitive_type_to_string(types::PrimitiveType::Int8), "Int8");
    EXPECT_EQ(types::primitive_type_to_string(types::PrimitiveType::Int16), "Int16");
    EXPECT_EQ(types::primitive_type_to_string(types::PrimitiveType::Int32), "Int32");
    EXPECT_EQ(types::primitive_type_to_string(types::PrimitiveType::Int64), "Int64");
    EXPECT_EQ(types::primitive_type_to_string(types::PrimitiveType::UInt8), "UInt8");
    EXPECT_EQ(types::primitive_type_to_string(types::PrimitiveType::UInt16), "UInt16");
    EXPECT_EQ(types::primitive_type_to_string(types::PrimitiveType::UInt32), "UInt32");
    EXPECT_EQ(types::primitive_type_to_string(types::PrimitiveType::UInt64), "UInt64");
    EXPECT_EQ(types::primitive_type_to_string(types::PrimitiveType::Float), "Float");
    EXPECT_EQ(types::primitive_type_to_string(types::PrimitiveType::Double), "Double");
}

TEST(PrimitiveTypeTest, FromString) {
    EXPECT_EQ(types::primitive_type_from_string("Void"), types::PrimitiveType::Void);
    EXPECT_EQ(types::primitive_type_from_string("Bool"), types::PrimitiveType::Bool);
    EXPECT_EQ(types::primitive_type_from_string("Int8"), types::PrimitiveType::Int8);
    EXPECT_EQ(types::primitive_type_from_string("Int16"), types::PrimitiveType::Int16);
    EXPECT_EQ(types::primitive_type_from_string("Int32"), types::PrimitiveType::Int32);
    EXPECT_EQ(types::primitive_type_from_string("Int64"), types::PrimitiveType::Int64);
    EXPECT_EQ(types::primitive_type_from_string("UInt8"), types::PrimitiveType::UInt8);
    EXPECT_EQ(types::primitive_type_from_string("UInt16"), types::PrimitiveType::UInt16);
    EXPECT_EQ(types::primitive_type_from_string("UInt32"), types::PrimitiveType::UInt32);
    EXPECT_EQ(types::primitive_type_from_string("UInt64"), types::PrimitiveType::UInt64);
    EXPECT_EQ(types::primitive_type_from_string("Float"), types::PrimitiveType::Float);
    EXPECT_EQ(types::primitive_type_from_string("Double"), types::PrimitiveType::Double);
}

TEST(PrimitiveTypeTest, BitWidth) {
    EXPECT_EQ(types::bit_width(types::PrimitiveType::Void), 0);
    EXPECT_EQ(types::bit_width(types::PrimitiveType::Bool), 1);
    EXPECT_EQ(types::bit_width(types::PrimitiveType::Int8), 8);
    EXPECT_EQ(types::bit_width(types::PrimitiveType::Int16), 16);
    EXPECT_EQ(types::bit_width(types::PrimitiveType::Int32), 32);
    EXPECT_EQ(types::bit_width(types::PrimitiveType::Int64), 64);
    EXPECT_EQ(types::bit_width(types::PrimitiveType::UInt8), 8);
    EXPECT_EQ(types::bit_width(types::PrimitiveType::UInt16), 16);
    EXPECT_EQ(types::bit_width(types::PrimitiveType::UInt32), 32);
    EXPECT_EQ(types::bit_width(types::PrimitiveType::UInt64), 64);
    EXPECT_EQ(types::bit_width(types::PrimitiveType::Float), 32);
    EXPECT_EQ(types::bit_width(types::PrimitiveType::Double), 64);
}

TEST(PrimitiveTypeTest, FloatingPoint) {
    EXPECT_TRUE(types::is_floating_point(types::PrimitiveType::Float));
    EXPECT_TRUE(types::is_floating_point(types::PrimitiveType::Double));
    EXPECT_FALSE(types::is_floating_point(types::PrimitiveType::Void));
    EXPECT_FALSE(types::is_floating_point(types::PrimitiveType::Bool));
    EXPECT_FALSE(types::is_floating_point(types::PrimitiveType::Int8));
    EXPECT_FALSE(types::is_floating_point(types::PrimitiveType::Int16));
    EXPECT_FALSE(types::is_floating_point(types::PrimitiveType::Int32));
    EXPECT_FALSE(types::is_floating_point(types::PrimitiveType::Int64));
    EXPECT_FALSE(types::is_floating_point(types::PrimitiveType::UInt8));
    EXPECT_FALSE(types::is_floating_point(types::PrimitiveType::UInt16));
    EXPECT_FALSE(types::is_floating_point(types::PrimitiveType::UInt32));
    EXPECT_FALSE(types::is_floating_point(types::PrimitiveType::UInt64));
}

TEST(PrimitiveTypeTest, Integer) {
    EXPECT_TRUE(types::is_integer(types::PrimitiveType::Bool));
    EXPECT_TRUE(types::is_integer(types::PrimitiveType::Int8));
    EXPECT_TRUE(types::is_integer(types::PrimitiveType::Int16));
    EXPECT_TRUE(types::is_integer(types::PrimitiveType::Int32));
    EXPECT_TRUE(types::is_integer(types::PrimitiveType::Int64));
    EXPECT_TRUE(types::is_integer(types::PrimitiveType::UInt8));
    EXPECT_TRUE(types::is_integer(types::PrimitiveType::UInt16));
    EXPECT_TRUE(types::is_integer(types::PrimitiveType::UInt32));
    EXPECT_TRUE(types::is_integer(types::PrimitiveType::UInt64));
    EXPECT_FALSE(types::is_integer(types::PrimitiveType::Void));
    EXPECT_FALSE(types::is_integer(types::PrimitiveType::Float));
    EXPECT_FALSE(types::is_integer(types::PrimitiveType::Double));
}

TEST(PrimitiveTypeTest, Signed) {
    EXPECT_TRUE(types::is_signed(types::PrimitiveType::Int8));
    EXPECT_TRUE(types::is_signed(types::PrimitiveType::Int16));
    EXPECT_TRUE(types::is_signed(types::PrimitiveType::Int32));
    EXPECT_TRUE(types::is_signed(types::PrimitiveType::Int64));
    EXPECT_FALSE(types::is_signed(types::PrimitiveType::Bool));
    EXPECT_FALSE(types::is_signed(types::PrimitiveType::UInt8));
    EXPECT_FALSE(types::is_signed(types::PrimitiveType::UInt16));
    EXPECT_FALSE(types::is_signed(types::PrimitiveType::UInt32));
    EXPECT_FALSE(types::is_signed(types::PrimitiveType::UInt64));
    EXPECT_FALSE(types::is_signed(types::PrimitiveType::Void));
    EXPECT_FALSE(types::is_signed(types::PrimitiveType::Float));
    EXPECT_FALSE(types::is_signed(types::PrimitiveType::Double));
}

TEST(PrimitiveTypeTest, Unsigned) {
    EXPECT_FALSE(types::is_unsigned(types::PrimitiveType::Int8));
    EXPECT_FALSE(types::is_unsigned(types::PrimitiveType::Int16));
    EXPECT_FALSE(types::is_unsigned(types::PrimitiveType::Int32));
    EXPECT_FALSE(types::is_unsigned(types::PrimitiveType::Int64));
    EXPECT_TRUE(types::is_unsigned(types::PrimitiveType::Bool));
    EXPECT_TRUE(types::is_unsigned(types::PrimitiveType::UInt8));
    EXPECT_TRUE(types::is_unsigned(types::PrimitiveType::UInt16));
    EXPECT_TRUE(types::is_unsigned(types::PrimitiveType::UInt32));
    EXPECT_TRUE(types::is_unsigned(types::PrimitiveType::UInt64));
    EXPECT_FALSE(types::is_unsigned(types::PrimitiveType::Void));
    EXPECT_FALSE(types::is_unsigned(types::PrimitiveType::Float));
    EXPECT_FALSE(types::is_unsigned(types::PrimitiveType::Double));
}

TEST(PrimitiveTypeTest, AsUnsigned) {
    EXPECT_EQ(types::as_unsigned(types::PrimitiveType::Int8), types::PrimitiveType::UInt8);
    EXPECT_EQ(types::as_unsigned(types::PrimitiveType::Int16), types::PrimitiveType::UInt16);
    EXPECT_EQ(types::as_unsigned(types::PrimitiveType::Int32), types::PrimitiveType::UInt32);
    EXPECT_EQ(types::as_unsigned(types::PrimitiveType::Int64), types::PrimitiveType::UInt64);
    EXPECT_EQ(types::as_unsigned(types::PrimitiveType::Bool), types::PrimitiveType::Bool);
    EXPECT_EQ(types::as_unsigned(types::PrimitiveType::UInt8), types::PrimitiveType::UInt8);
    EXPECT_EQ(types::as_unsigned(types::PrimitiveType::UInt16), types::PrimitiveType::UInt16);
    EXPECT_EQ(types::as_unsigned(types::PrimitiveType::UInt32), types::PrimitiveType::UInt32);
    EXPECT_EQ(types::as_unsigned(types::PrimitiveType::UInt64), types::PrimitiveType::UInt64);
    EXPECT_EQ(types::as_unsigned(types::PrimitiveType::Void), types::PrimitiveType::Void);
    EXPECT_EQ(types::as_unsigned(types::PrimitiveType::Float), types::PrimitiveType::Float);
    EXPECT_EQ(types::as_unsigned(types::PrimitiveType::Double), types::PrimitiveType::Double);
}

TEST(PrimitiveTypeTest, AsSigned) {
    EXPECT_EQ(types::as_signed(types::PrimitiveType::Int8), types::PrimitiveType::Int8);
    EXPECT_EQ(types::as_signed(types::PrimitiveType::Int16), types::PrimitiveType::Int16);
    EXPECT_EQ(types::as_signed(types::PrimitiveType::Int32), types::PrimitiveType::Int32);
    EXPECT_EQ(types::as_signed(types::PrimitiveType::Int64), types::PrimitiveType::Int64);
    EXPECT_EQ(types::as_signed(types::PrimitiveType::Bool), types::PrimitiveType::Bool);
    EXPECT_EQ(types::as_signed(types::PrimitiveType::UInt8), types::PrimitiveType::Int8);
    EXPECT_EQ(types::as_signed(types::PrimitiveType::UInt16), types::PrimitiveType::Int16);
    EXPECT_EQ(types::as_signed(types::PrimitiveType::UInt32), types::PrimitiveType::Int32);
    EXPECT_EQ(types::as_signed(types::PrimitiveType::UInt64), types::PrimitiveType::Int64);
    EXPECT_EQ(types::as_signed(types::PrimitiveType::Void), types::PrimitiveType::Void);
    EXPECT_EQ(types::as_signed(types::PrimitiveType::Float), types::PrimitiveType::Float);
    EXPECT_EQ(types::as_signed(types::PrimitiveType::Double), types::PrimitiveType::Double);
}
