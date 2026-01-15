#include "sdfg/types/array.h"

#include <gtest/gtest.h>

#include "sdfg/types/scalar.h"

using namespace sdfg;

TEST(ArrayTest, Init) {
    types::Scalar s(types::PrimitiveType::Int32);
    symbolic::Expression e = symbolic::integer(10);

    types::Array a(s, e);
    EXPECT_EQ(a.element_type(), s);
    EXPECT_TRUE(symbolic::eq(a.num_elements(), e));
    EXPECT_EQ(a.primitive_type(), types::PrimitiveType::Int32);
}

TEST(ArrayTest, Symbol) {
    types::Scalar s(types::PrimitiveType::UInt32);
    symbolic::Expression e = symbolic::integer(10);

    types::Array a(s, e);
    EXPECT_FALSE(a.is_symbol());
}

TEST(ArrayTest, TypeId) {
    types::Scalar s(types::PrimitiveType::Float);
    symbolic::Expression e = symbolic::integer(10);

    types::Array a(s, e);
    EXPECT_EQ(a.type_id(), types::TypeID::Array);
}

TEST(ArrayTest, Equal) {
    types::Scalar s(types::PrimitiveType::UInt32);
    symbolic::Expression e = symbolic::integer(10);

    types::Array a(s, e);
    types::Array a2(s, e);
    types::Scalar s2(types::PrimitiveType::Float);
    symbolic::Expression e2 = symbolic::integer(20);
    types::Array a3(s2, e2);
    EXPECT_EQ(a, a2);
    EXPECT_NE(a, a3);
}

TEST(ArrayTest, Clone) {
    types::Scalar s(types::PrimitiveType::Float);
    symbolic::Expression e = symbolic::integer(10);

    types::Array a(s, e);
    auto a2 = a.clone();

    auto a2_ = dynamic_cast<types::Array*>(a2.get());

    EXPECT_EQ(a.element_type(), a2_->element_type());
    EXPECT_TRUE(symbolic::eq(a.num_elements(), a2_->num_elements()));
}
