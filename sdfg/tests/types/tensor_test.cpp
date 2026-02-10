#include "sdfg/types/tensor.h"

#include <gtest/gtest.h>

#include "sdfg/types/scalar.h"

using namespace sdfg;

TEST(TensorTest, Init) {
    types::Scalar s(types::PrimitiveType::Int32);
    symbolic::MultiExpression shape = {symbolic::integer(10), symbolic::integer(20)};

    types::Tensor a(s, shape);

    EXPECT_EQ(a.element_type(), s);
    EXPECT_EQ(a.primitive_type(), types::PrimitiveType::Int32);

    EXPECT_EQ(a.shape().size(), 2);
    EXPECT_TRUE(symbolic::eq(a.shape().at(0), symbolic::integer(10)));
    EXPECT_TRUE(symbolic::eq(a.shape().at(1), symbolic::integer(20)));

    EXPECT_EQ(a.strides().size(), 2);
    EXPECT_TRUE(symbolic::eq(a.strides().at(0), symbolic::mul(symbolic::integer(20), symbolic::integer(1))));
    EXPECT_TRUE(symbolic::eq(a.strides().at(1), symbolic::integer(1)));

    EXPECT_FALSE(a.is_symbol());
    EXPECT_EQ(a.type_id(), types::TypeID::Tensor);
}

TEST(TensorTest, Equal) {
    types::Scalar s(types::PrimitiveType::UInt32);
    symbolic::MultiExpression shape = {symbolic::integer(10), symbolic::integer(20)};

    types::Tensor a(s, shape);
    types::Tensor a2(s, shape);

    types::Scalar s2(types::PrimitiveType::Float);
    types::Tensor a4(s2, shape);

    symbolic::MultiExpression shape2 = {symbolic::integer(10), symbolic::integer(30)};
    types::Tensor a3(s, shape2);

    EXPECT_EQ(a, a2);
    EXPECT_NE(a, a3);
    EXPECT_NE(a, a4);
}

TEST(TensorTest, Clone) {
    types::Scalar s(types::PrimitiveType::Float);
    symbolic::MultiExpression shape = {symbolic::integer(10), symbolic::integer(20)};

    types::Tensor a(s, shape);
    auto a2 = a.clone();

    auto a2_ = dynamic_cast<types::Tensor*>(a2.get());

    EXPECT_EQ(a.element_type(), a2_->element_type());
    EXPECT_EQ(a.shape().size(), a2_->shape().size());
    EXPECT_TRUE(symbolic::eq(a.shape().at(0), a2_->shape().at(0)));
    EXPECT_TRUE(symbolic::eq(a.shape().at(1), a2_->shape().at(1)));
}
