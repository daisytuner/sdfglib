#include "sdfg/codegen/code_generators/cpp_code_generator.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"

using namespace sdfg;

TEST(CPPCodeGeneratorTest, FunctionDefintion) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto sdfg = builder.move();

    codegen::CPPCodeGenerator generator(*sdfg);
    auto result = generator.function_definition();
    EXPECT_EQ(result, "extern \"C\" void sdfg_a()");
}

TEST(CPPCodeGeneratorTest, Dispatch_Includes) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto sdfg = builder.move();

    codegen::CPPCodeGenerator generator(*sdfg);
    EXPECT_TRUE(generator.generate());

    auto result = generator.includes().str();
    EXPECT_EQ(
        result,
        "#include <cmath>\n#include <cblas.h>\n#define __daisy_min(a,b) ((a)<(b)?(a):(b))\n#define "
        "__daisy_max(a,b) ((a)>(b)?(a):(b))\n#define __daisy_fma(a,b,c) a * b + c\n"
    );
}

TEST(CPPCodeGeneratorTest, DispatchStructures_Basic) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);

    auto& struct_def_A = builder.add_structure("MyStructA", false);
    struct_def_A.add_member(types::Scalar(types::PrimitiveType::UInt8));

    auto sdfg = builder.move();

    codegen::CPPCodeGenerator generator(*sdfg);
    EXPECT_TRUE(generator.generate());

    auto result = generator.classes().str();
    EXPECT_EQ(result, R"(struct MyStructA;
struct MyStructA
{
char member_0;
};
)");
}

TEST(CPPCodeGeneratorTest, DispatchStructures_Packed) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);

    auto& struct_def_A = builder.add_structure("MyStructA", true);
    struct_def_A.add_member(types::Scalar(types::PrimitiveType::UInt8));

    auto sdfg = builder.move();

    codegen::CPPCodeGenerator generator(*sdfg);
    EXPECT_TRUE(generator.generate());

    auto result = generator.classes().str();
    EXPECT_EQ(result, R"(struct MyStructA;
struct __attribute__((packed)) MyStructA
{
char member_0;
};
)");
}

TEST(CPPCodeGeneratorTest, DispatchStructures_Nested) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);

    auto& struct_def_A = builder.add_structure("MyStructA", false);
    struct_def_A.add_member(types::Scalar(types::PrimitiveType::UInt8));

    auto& struct_def_B = builder.add_structure("MyStructB", false);
    struct_def_B.add_member(types::Structure("MyStructA"));

    auto sdfg = builder.move();

    codegen::CPPCodeGenerator generator(*sdfg);
    EXPECT_TRUE(generator.generate());

    auto result = generator.classes().str();
    EXPECT_EQ(result, R"(struct MyStructB;
struct MyStructA;
struct MyStructA
{
char member_0;
};
struct MyStructB
{
struct MyStructA member_0;
};
)");
}

TEST(CPPCodeGeneratorTest, DispatchGlobals) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);

    builder.add_container("a", types::Scalar(types::PrimitiveType::Int32), false, true);

    auto sdfg = builder.move();

    codegen::CPPCodeGenerator generator(*sdfg);
    EXPECT_TRUE(generator.generate());

    auto result = generator.globals().str();
    EXPECT_EQ(result, "extern int a;\n");
}
