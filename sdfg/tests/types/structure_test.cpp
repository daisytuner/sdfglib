#include "sdfg/types/structure.h"

#include <gtest/gtest.h>

#include "sdfg/types/scalar.h"

using namespace sdfg;

TEST(StructureTest, Init) {
    types::Structure s("test");
    EXPECT_EQ(s.name(), "test");
    EXPECT_EQ(s.primitive_type(), types::PrimitiveType::Void);
}

TEST(StructureTest, Symbol) {
    types::Structure s("test");
    EXPECT_FALSE(s.is_symbol());
}

TEST(StructureTest, Equal) {
    types::Structure s("test");
    types::Structure s2("test");
    types::Structure s3("test2");
    EXPECT_EQ(s, s2);
    EXPECT_NE(s, s3);
}

TEST(StructureTest, Clone) {
    types::Structure s("test");
    auto s2 = s.clone();
    auto s2_ = dynamic_cast<types::Structure*>(s2.get());
    EXPECT_EQ(s.name(), s2_->name());
}
TEST(StructureTest, TypeId) {
    types::Structure s("test");
    EXPECT_EQ(s.type_id(), types::TypeID::Structure);
}

TEST(StructureTest, StructureDefinition) {
    types::StructureDefinition s("test", false);
    EXPECT_EQ(s.name(), "test");
    EXPECT_FALSE(s.is_packed());

    s.add_member(types::Scalar(types::PrimitiveType::Int32));
    s.add_member(types::Scalar(types::PrimitiveType::Int64));
    EXPECT_EQ(s.num_members(), 2);

    auto& member_1 = s.member_type(symbolic::integer(0));
    auto& member_2 = s.member_type(symbolic::integer(1));
    EXPECT_TRUE(dynamic_cast<const types::Scalar*>(&member_1));
    EXPECT_TRUE(dynamic_cast<const types::Scalar*>(&member_2));
    EXPECT_EQ(member_1.primitive_type(), types::PrimitiveType::Int32);
    EXPECT_EQ(member_2.primitive_type(), types::PrimitiveType::Int64);
}

TEST(StructureTest, StructureDefinition_Clone) {
    types::StructureDefinition s("test", true);
    s.add_member(types::Scalar(types::PrimitiveType::Int32));
    s.add_member(types::Scalar(types::PrimitiveType::Int64));

    auto s2 = s.clone();
    auto s2_ = s2.get();

    EXPECT_EQ(s.name(), s2_->name());
    EXPECT_EQ(s.is_packed(), s2_->is_packed());
    EXPECT_TRUE(s.is_packed());
    EXPECT_EQ(s.num_members(), s2_->num_members());

    auto& member_1 = s.member_type(symbolic::integer(0));
    auto& member_2 = s.member_type(symbolic::integer(1));
    auto& member_1_ = s2_->member_type(symbolic::integer(0));
    auto& member_2_ = s2_->member_type(symbolic::integer(1));

    EXPECT_TRUE(dynamic_cast<const types::Scalar*>(&member_1));
    EXPECT_TRUE(dynamic_cast<const types::Scalar*>(&member_2));
    EXPECT_TRUE(dynamic_cast<const types::Scalar*>(&member_1_));
    EXPECT_TRUE(dynamic_cast<const types::Scalar*>(&member_2_));
    EXPECT_EQ(member_1, member_1_);
    EXPECT_EQ(member_2, member_2_);
}
