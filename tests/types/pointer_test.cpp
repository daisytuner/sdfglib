#include "sdfg/types/pointer.h"

#include <gtest/gtest.h>

#include "sdfg/types/scalar.h"

using namespace sdfg;

TEST(PointerTest, Init) {
    types::Pointer p(types::Scalar(types::PrimitiveType::Int32));
    EXPECT_EQ(p.pointee_type(), types::Scalar(types::PrimitiveType::Int32));
    EXPECT_EQ(p.primitive_type(), types::PrimitiveType::Int32);
}

TEST(PointerTest, Symbol) {
    types::Pointer p(types::Scalar(types::PrimitiveType::UInt32));
    EXPECT_TRUE(p.is_symbol());

    types::Pointer p2(types::Scalar(types::PrimitiveType::Float));
    EXPECT_TRUE(p2.is_symbol());
}

TEST(PointerTest, Equal) {
    types::Pointer p(types::Scalar(types::PrimitiveType::UInt32));
    types::Pointer p2(types::Scalar(types::PrimitiveType::UInt32));
    types::Pointer p3(types::Scalar(types::PrimitiveType::Float));
    EXPECT_EQ(p, p2);
    EXPECT_NE(p, p3);
}

TEST(PointerTest, Clone) {
    types::Pointer p(types::Scalar(types::PrimitiveType::Float));
    auto p2 = p.clone();

    auto p2_ = dynamic_cast<types::Pointer*>(p2.get());
    EXPECT_EQ(p.pointee_type(), p2_->pointee_type());
}
