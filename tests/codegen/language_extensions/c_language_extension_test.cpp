#include "sdfg/codegen/language_extensions/c_language_extension.h"

#include <gtest/gtest.h>

#include "sdfg/builder/sdfg_builder.h"

using namespace sdfg;

TEST(CLanguageExtensionTest, PrimitiveType_Void) {
    codegen::CLanguageExtension generator;
    auto result = generator.primitive_type(types::PrimitiveType::Void);
    EXPECT_EQ(result, "void");
}

TEST(CLanguageExtensionTest, PrimitiveType_Bool) {
    codegen::CLanguageExtension generator;
    auto result = generator.primitive_type(types::PrimitiveType::Bool);
    EXPECT_EQ(result, "bool");
}

TEST(CLanguageExtensionTest, PrimitiveType_Int8) {
    codegen::CLanguageExtension generator;
    auto result = generator.primitive_type(types::PrimitiveType::Int8);
    EXPECT_EQ(result, "signed char");
}

TEST(CLanguageExtensionTest, PrimitiveType_Int16) {
    codegen::CLanguageExtension generator;
    auto result = generator.primitive_type(types::PrimitiveType::Int16);
    EXPECT_EQ(result, "short");
}

TEST(CLanguageExtensionTest, PrimitiveType_Int32) {
    codegen::CLanguageExtension generator;
    auto result = generator.primitive_type(types::PrimitiveType::Int32);
    EXPECT_EQ(result, "int");
}

TEST(CLanguageExtensionTest, PrimitiveType_Int64) {
    codegen::CLanguageExtension generator;
    auto result = generator.primitive_type(types::PrimitiveType::Int64);
    EXPECT_EQ(result, "long long");
}

TEST(CLanguageExtensionTest, PrimitiveType_UInt8) {
    codegen::CLanguageExtension generator;
    auto result = generator.primitive_type(types::PrimitiveType::UInt8);
    EXPECT_EQ(result, "char");
}

TEST(CLanguageExtensionTest, PrimitiveType_UInt16) {
    codegen::CLanguageExtension generator;
    auto result = generator.primitive_type(types::PrimitiveType::UInt16);
    EXPECT_EQ(result, "unsigned short");
}

TEST(CLanguageExtensionTest, PrimitiveType_UInt32) {
    codegen::CLanguageExtension generator;
    auto result = generator.primitive_type(types::PrimitiveType::UInt32);
    EXPECT_EQ(result, "unsigned int");
}

TEST(CLanguageExtensionTest, PrimitiveType_UInt64) {
    codegen::CLanguageExtension generator;
    auto result = generator.primitive_type(types::PrimitiveType::UInt64);
    EXPECT_EQ(result, "unsigned long long");
}

TEST(CLanguageExtensionTest, PrimitiveType_Float) {
    codegen::CLanguageExtension generator;
    auto result = generator.primitive_type(types::PrimitiveType::Float);
    EXPECT_EQ(result, "float");
}

TEST(CLanguageExtensionTest, PrimitiveType_Double) {
    codegen::CLanguageExtension generator;
    auto result = generator.primitive_type(types::PrimitiveType::Double);
    EXPECT_EQ(result, "double");
}

TEST(CLanguageExtensionTest, Declaration_Scalar) {
    codegen::CLanguageExtension generator;
    auto result = generator.declaration("var", types::Scalar(types::PrimitiveType::Int32));
    EXPECT_EQ(result, "int var");
}

TEST(CLanguageExtensionTest, Declaration_Pointer) {
    codegen::CLanguageExtension generator;
    auto result =
        generator.declaration("var", types::Pointer(types::Scalar(types::PrimitiveType::Int32)));
    EXPECT_EQ(result, "int *var");
}

TEST(CLanguageExtensionTest, Declaration_Array) {
    codegen::CLanguageExtension generator;
    auto result = generator.declaration(
        "var", types::Array(types::Scalar(types::PrimitiveType::Int32), symbolic::integer(10)));
    EXPECT_EQ(result, "int var[10]");
}

TEST(CLanguageExtensionTest, Declaration_Struct) {
    codegen::CLanguageExtension generator;
    auto result = generator.declaration("var", types::Structure("MyStruct"));
    EXPECT_EQ(result, "MyStruct var");
}

TEST(CLanguageExtensionTest, Declaration_ArrayOfStruct) {
    codegen::CLanguageExtension generator;
    auto result = generator.declaration(
        "var", types::Array(types::Structure("MyStruct"), symbolic::integer(10)));
    EXPECT_EQ(result, "MyStruct var[10]");
}

TEST(CLanguageExtensionTest, Declaration_PointerToArray) {
    codegen::CLanguageExtension generator;
    auto result = generator.declaration(
        "var", types::Pointer(types::Array(types::Scalar(types::PrimitiveType::Int32),
                                           symbolic::integer(10))));
    EXPECT_EQ(result, "int (*var)[10]");
}

TEST(CLanguageExtensionTest, Allocation_Scalar) {
    codegen::CLanguageExtension generator;
    auto result = generator.allocation("var", types::Scalar(types::PrimitiveType::Int32));
    EXPECT_EQ(result, "int var");
}

TEST(CLanguageExtensionTest, Allocation_Array) {
    codegen::CLanguageExtension generator;
    auto result = generator.allocation(
        "var", types::Array(types::Scalar(types::PrimitiveType::Int32), symbolic::integer(10)));
    EXPECT_EQ(result, "int var[10]");
}

TEST(CLanguageExtensionTest, Allocation_Pointer) {
    codegen::CLanguageExtension generator;
    auto result =
        generator.allocation("var", types::Pointer(types::Scalar(types::PrimitiveType::Float)));
    EXPECT_EQ(result, "float *var = (float *) malloc(1 * sizeof(float ))");
}

TEST(CLanguageExtensionTest, Allocation_Struct) {
    codegen::CLanguageExtension generator;
    auto result = generator.allocation("var", types::Structure("MyStruct"));
    EXPECT_EQ(result, "MyStruct var");
}

TEST(CLanguageExtensionTest, Deallocation_Scalar) {
    codegen::CLanguageExtension generator;
    auto result = generator.deallocation("var", types::Scalar(types::PrimitiveType::Int32));
    EXPECT_EQ(result, "");
}

TEST(CLanguageExtensionTest, Deallocation_Array) {
    codegen::CLanguageExtension generator;
    auto result = generator.deallocation(
        "var", types::Array(types::Scalar(types::PrimitiveType::Int32), symbolic::integer(10)));
    EXPECT_EQ(result, "");
}

TEST(CLanguageExtensionTest, Deallocation_Pointer) {
    codegen::CLanguageExtension generator;
    auto result =
        generator.deallocation("var", types::Pointer(types::Scalar(types::PrimitiveType::Float)));
    EXPECT_EQ(result, "free(var)");
}

TEST(CLanguageExtensionTest, Deallocation_Struct) {
    codegen::CLanguageExtension generator;
    auto result = generator.deallocation("var", types::Structure("MyStruct"));
    EXPECT_EQ(result, "");
}

TEST(CLanguageExtensionTest, Typecast) {
    codegen::CLanguageExtension generator;
    auto result =
        generator.type_cast("var", types::Pointer(types::Scalar(types::PrimitiveType::Float)));
    EXPECT_EQ(result, "(float *) var");
}

TEST(CLanguageExtensionTest, SubsetToCpp_Scalar) {
    builder::SDFGBuilder builder("sdfg");
    auto& sdfg = builder.subject();

    codegen::CLanguageExtension generator;
    auto result =
        generator.subset(sdfg, types::Scalar(types::PrimitiveType::Int32), data_flow::Subset());
    EXPECT_EQ(result, "");
}

TEST(CLanguageExtensionTest, SubsetToCpp_Array) {
    builder::SDFGBuilder builder("sdfg");
    auto& sdfg = builder.subject();

    codegen::CLanguageExtension generator;
    auto result = generator.subset(
        sdfg, types::Array(types::Scalar(types::PrimitiveType::Int32), symbolic::integer(10)),
        data_flow::Subset{symbolic::integer(1)});
    EXPECT_EQ(result, "[1]");
}

TEST(CLanguageExtensionTest, SubsetToCpp_Struct) {
    builder::SDFGBuilder builder("sdfg");
    auto& sdfg = builder.subject();

    auto& struct_def = builder.add_structure("MyStruct");
    struct_def.add_member(types::Scalar(types::PrimitiveType::Int32));
    struct_def.add_member(types::Scalar(types::PrimitiveType::Float));

    codegen::CLanguageExtension generator;
    auto result = generator.subset(sdfg, types::Structure("MyStruct"),
                                   data_flow::Subset{symbolic::integer(1)});
    EXPECT_EQ(result, ".member_1");
}
