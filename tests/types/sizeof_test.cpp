#include "sdfg/types/utils.h"

#include <gtest/gtest.h>

#include "sdfg/builder/sdfg_builder.h"

using namespace sdfg;

TEST(SizeOfTest, StaticElementSizeOf2DArrayLike) {
    auto type = types::Pointer(types::Array(types::Scalar(types::Float), symbolic::integer(1000)));

    auto s = types::get_contiguous_element_size(type, false);

    EXPECT_TRUE(symbolic::eq(s, symbolic::integer(4)));
}

TEST(SizeOfTest, StaticSizeOfPointer) {
    auto innerPtrType = types::Pointer(types::Scalar(types::Float));
    auto type = types::Pointer(types::Array(innerPtrType, symbolic::integer(1000)));

    auto s = types::get_contiguous_element_size(type, false); // ptr-size

    auto ptr_size = types::get_type_size(innerPtrType, false);

    EXPECT_TRUE(symbolic::eq(s, symbolic::integer(8)));
    EXPECT_TRUE(symbolic::eq(ptr_size, symbolic::integer(8)));
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

TEST(SizeOfTest, NoStaticSizeOfStruct) {
    auto type = types::Structure("some_t");

    auto s = types::get_type_size(type, false);

    EXPECT_TRUE(s.is_null());
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

TEST(MallocUsableSizeTest, MallocUsableSizeSymbol) {
    auto symbol = symbolic::symbol("my_array");
    auto expr = symbolic::malloc_usable_size(symbol);

    auto f = SymEngine::rcp_dynamic_cast<const sdfg::symbolic::MallocUsableSizeFunction>(expr);
    auto args = f->get_args();
    auto s = args[0];
    EXPECT_TRUE(symbolic::eq(s, symbol));


    auto symbol2 = symbolic::symbol("my_array");
    auto expr2 = symbolic::malloc_usable_size(symbol2);
    EXPECT_TRUE(symbolic::eq(expr, expr2));

    auto symbol3 = symbolic::symbol("my_array3");
    auto expr3 = symbolic::malloc_usable_size(symbol3);
    EXPECT_FALSE(symbolic::eq(expr, expr3));

    EXPECT_EQ(expr->__str__(), "malloc_usable_size(my_array)");
}
