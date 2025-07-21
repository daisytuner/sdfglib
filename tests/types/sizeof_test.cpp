#include "sdfg/types/utils.h"

#include <gtest/gtest.h>

#include "sdfg/builder/sdfg_builder.h"

using namespace sdfg;

TEST(SizeOfTest, StaticElementSizeOf2DArrayLike) {
    auto type = types::Pointer(types::Array(types::Scalar(types::Float), symbolic::integer(1000)));

    auto s = types::get_contiguous_element_size(type, false);

    EXPECT_TRUE(symbolic::eq(s, symbolic::integer(4)));
}

TEST(SizeOfTest, SizeOfPointer) {
    auto innerPtrType = types::Pointer(types::Scalar(types::Float));
    auto type = types::Pointer(types::Array(innerPtrType, symbolic::integer(1000)));

    auto s = types::get_contiguous_element_size(type, false); // ptr-size

    auto ptr_size = types::get_type_size(innerPtrType, false);

    EXPECT_EQ(s.is_null(), ptr_size.is_null());

    if (!s.is_null()) {
        EXPECT_TRUE(symbolic::eq(s, ptr_size));
    }
}

TEST(SizeOfTest, StaticSizeOfArray) {
    auto type = types::Array(types::Scalar(types::Float), symbolic::integer(1000));

    auto s = types::get_type_size(type, false);

    EXPECT_TRUE(symbolic::eq(s, symbolic::integer(4000)));
}

TEST(SizeOfTest, StaticSizeOfScalar) {
    auto type = types::Scalar(types::Int16);

    auto s = types::get_type_size(type, false);

    EXPECT_TRUE(symbolic::eq(s, symbolic::integer(2)));
}

TEST(SizeOfTest, NoStaticSizeOfPointer) {
    auto type = types::Pointer(types::Scalar(types::Int16));

    auto s = types::get_type_size(type, false);

    EXPECT_TRUE(s.is_null());
}

TEST(SizeOfTest, NoStaticSizeOfStruct) {
    auto type = types::Structure("some_t");

    auto s = types::get_type_size(type, false);

    EXPECT_TRUE(s.is_null());
}

TEST(SizeOfTest, SymbolicSizeOfPointer) {
    auto type = types::Pointer(types::Scalar(types::Int16));

    auto s = types::get_type_size(type, true);

    EXPECT_TRUE(symbolic::eq(s, symbolic::size_of_type(type)));
    auto f = SymEngine::rcp_dynamic_cast<const symbolic::SizeOfTypeFunction>(s);
    auto& t = f->get_type();
    EXPECT_EQ(t, type);
}

TEST(SizeOfTest, SymbolicSizeOfStruct) {
    auto type = types::Structure("some_t");

    auto s = types::get_type_size(type, true);

    EXPECT_TRUE(symbolic::eq(s, symbolic::size_of_type(type)));
    auto f = SymEngine::rcp_dynamic_cast<const symbolic::SizeOfTypeFunction>(s);
    auto& t = f->get_type();
    EXPECT_EQ(t, type);
}

TEST(SizeOfTest, SymbolicSizeOfNestedStruct) {
    auto struct_type = types::Structure("some_t");
    auto type = types::Pointer(types::Array(struct_type, symbolic::integer(1000)));

    auto s = types::get_contiguous_element_size(type, true);

    EXPECT_TRUE(symbolic::eq(s, symbolic::size_of_type(type)));
    auto f = SymEngine::rcp_dynamic_cast<const symbolic::SizeOfTypeFunction>(s);
    auto& t = f->get_type();
    EXPECT_EQ(t.type_id(), types::TypeID::Structure);
    EXPECT_EQ(t, struct_type);
}
