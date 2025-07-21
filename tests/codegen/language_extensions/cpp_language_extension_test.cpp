#include "sdfg/codegen/language_extensions/cpp_language_extension.h"

#include <gtest/gtest.h>

#include "sdfg/builder/sdfg_builder.h"
#include "sdfg/types/utils.h"

using namespace sdfg;

TEST(CPPLanguageExtensionTest, PrimitiveType_Void) {
    codegen::CPPLanguageExtension generator;
    auto result = generator.primitive_type(types::PrimitiveType::Void);
    EXPECT_EQ(result, "void");
}

TEST(CPPLanguageExtensionTest, PrimitiveType_Bool) {
    codegen::CPPLanguageExtension generator;
    auto result = generator.primitive_type(types::PrimitiveType::Bool);
    EXPECT_EQ(result, "bool");
}

TEST(CPPLanguageExtensionTest, PrimitiveType_Int8) {
    codegen::CPPLanguageExtension generator;
    auto result = generator.primitive_type(types::PrimitiveType::Int8);
    EXPECT_EQ(result, "signed char");
}

TEST(CPPLanguageExtensionTest, PrimitiveType_Int16) {
    codegen::CPPLanguageExtension generator;
    auto result = generator.primitive_type(types::PrimitiveType::Int16);
    EXPECT_EQ(result, "short");
}

TEST(CPPLanguageExtensionTest, PrimitiveType_Int32) {
    codegen::CPPLanguageExtension generator;
    auto result = generator.primitive_type(types::PrimitiveType::Int32);
    EXPECT_EQ(result, "int");
}

TEST(CPPLanguageExtensionTest, PrimitiveType_Int64) {
    codegen::CPPLanguageExtension generator;
    auto result = generator.primitive_type(types::PrimitiveType::Int64);
    EXPECT_EQ(result, "long long");
}

TEST(CPPLanguageExtensionTest, PrimitiveType_UInt8) {
    codegen::CPPLanguageExtension generator;
    auto result = generator.primitive_type(types::PrimitiveType::UInt8);
    EXPECT_EQ(result, "char");
}

TEST(CPPLanguageExtensionTest, PrimitiveType_UInt16) {
    codegen::CPPLanguageExtension generator;
    auto result = generator.primitive_type(types::PrimitiveType::UInt16);
    EXPECT_EQ(result, "unsigned short");
}

TEST(CPPLanguageExtensionTest, PrimitiveType_UInt32) {
    codegen::CPPLanguageExtension generator;
    auto result = generator.primitive_type(types::PrimitiveType::UInt32);
    EXPECT_EQ(result, "unsigned int");
}

TEST(CPPLanguageExtensionTest, PrimitiveType_UInt64) {
    codegen::CPPLanguageExtension generator;
    auto result = generator.primitive_type(types::PrimitiveType::UInt64);
    EXPECT_EQ(result, "unsigned long long");
}

TEST(CPPLanguageExtensionTest, PrimitiveType_Float) {
    codegen::CPPLanguageExtension generator;
    auto result = generator.primitive_type(types::PrimitiveType::Float);
    EXPECT_EQ(result, "float");
}

TEST(CPPLanguageExtensionTest, PrimitiveType_Double) {
    codegen::CPPLanguageExtension generator;
    auto result = generator.primitive_type(types::PrimitiveType::Double);
    EXPECT_EQ(result, "double");
}

TEST(CPPLanguageExtensionTest, Declaration_Scalar) {
    codegen::CPPLanguageExtension generator;
    auto result = generator.declaration("var", types::Scalar(types::PrimitiveType::Int32));
    EXPECT_EQ(result, "int var");
}

TEST(CPPLanguageExtensionTest, Declaration_Pointer) {
    codegen::CPPLanguageExtension generator;
    auto result = generator.declaration("var", types::Pointer(types::Scalar(types::PrimitiveType::Int32)));
    EXPECT_EQ(result, "int *var");
}

TEST(CPPLanguageExtensionTest, Declaration_Array) {
    codegen::CPPLanguageExtension generator;
    auto result =
        generator.declaration("var", types::Array(types::Scalar(types::PrimitiveType::Int32), symbolic::integer(10)));
    EXPECT_EQ(result, "int var[10]");
}

TEST(CPPLanguageExtensionTest, Declaration_Struct) {
    codegen::CPPLanguageExtension generator;
    auto result = generator.declaration("var", types::Structure("MyStruct"));
    EXPECT_EQ(result, "MyStruct var");
}

TEST(CPPLanguageExtensionTest, Declaration_ArrayOfStruct) {
    codegen::CPPLanguageExtension generator;
    auto result = generator.declaration("var", types::Array(types::Structure("MyStruct"), symbolic::integer(10)));
    EXPECT_EQ(result, "MyStruct var[10]");
}

TEST(CPPLanguageExtensionTest, Declaration_PointerToArray) {
    codegen::CPPLanguageExtension generator;
    auto result = generator.declaration(
        "var", types::Pointer(types::Array(types::Scalar(types::PrimitiveType::Int32), symbolic::integer(10)))
    );
    EXPECT_EQ(result, "int (*var)[10]");
}

TEST(CPPLanguageExtensionTest, Typecast) {
    codegen::CPPLanguageExtension generator;
    auto result = generator.type_cast("var", types::Pointer(types::Scalar(types::PrimitiveType::Float)));
    EXPECT_EQ(result, "reinterpret_cast<float *>(var)");
}

TEST(CPPLanguageExtensionTest, Sizeof) {
    codegen::CPPLanguageExtension generator;
    auto type = types::Pointer(types::Structure("some_t"));
    auto size_expr = types::get_contiguous_element_size(type, true);
    auto result = generator.expression(size_expr);
    EXPECT_EQ(result, "sizeof(some_t )");
}

TEST(CPPLanguageExtensionTest, SubsetToCpp_Scalar) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    codegen::CPPLanguageExtension generator;
    auto result = generator.subset(sdfg, types::Scalar(types::PrimitiveType::Int32), data_flow::Subset());
    EXPECT_EQ(result, "");
}

TEST(CPPLanguageExtensionTest, SubsetToCpp_Array) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    codegen::CPPLanguageExtension generator;
    auto result = generator.subset(
        sdfg,
        types::Array(types::Scalar(types::PrimitiveType::Int32), symbolic::integer(10)),
        data_flow::Subset{symbolic::integer(1)}
    );
    EXPECT_EQ(result, "[1]");
}

TEST(CPPLanguageExtensionTest, SubsetToCpp_Struct) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    auto& struct_def = builder.add_structure("MyStruct", false);
    struct_def.add_member(types::Scalar(types::PrimitiveType::Int32));
    struct_def.add_member(types::Scalar(types::PrimitiveType::Float));

    codegen::CPPLanguageExtension generator;
    auto result = generator.subset(sdfg, types::Structure("MyStruct"), data_flow::Subset{symbolic::integer(1)});
    EXPECT_EQ(result, ".member_1");
}
