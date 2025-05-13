#include "sdfg/codegen/code_generators/c_code_generator.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/conditional_schedule.h"

using namespace sdfg;

TEST(CCodeGeneratorTest, FunctionDefintion) {
    builder::StructuredSDFGBuilder builder("sdfg_a");
    auto sdfg = builder.move();

    ConditionalSchedule schedule(sdfg);

    codegen::CCodeGenerator generator(schedule, false);
    auto result = generator.function_definition();
    EXPECT_EQ(result, "extern void sdfg_a()");
}

TEST(CCodeGeneratorTest, Dispatch_Includes) {
    builder::StructuredSDFGBuilder builder("sdfg_a");
    auto sdfg = builder.move();

    ConditionalSchedule schedule(sdfg);

    codegen::CCodeGenerator generator(schedule, false);
    EXPECT_TRUE(generator.generate());

    auto result = generator.includes().str();
    EXPECT_EQ(result,
              "#include <math.h>\n#include <stdbool.h>\n#include <stdlib.h>\n#define "
              "__daisy_min(a,b) ((a)<(b)?(a):(b))\n#define __daisy_max(a,b) "
              "((a)>(b)?(a):(b))\n#define __daisy_fma(a,b,c) a * b + c\n");
}

TEST(CCodeGeneratorTest, DispatchStructures_Basic) {
    builder::StructuredSDFGBuilder builder("sdfg_a");

    auto& struct_def_A = builder.add_structure("MyStructA", false);
    struct_def_A.add_member(types::Scalar(types::PrimitiveType::UInt8));

    auto sdfg = builder.move();

    ConditionalSchedule schedule(sdfg);

    codegen::CCodeGenerator generator(schedule, false);
    EXPECT_TRUE(generator.generate());

    auto result = generator.classes().str();
    EXPECT_EQ(result, R"(typedef struct MyStructA MyStructA;
typedef struct MyStructA
{
char member_0;
} MyStructA;
)");
}

TEST(CCodeGeneratorTest, DispatchStructures_Nested) {
    builder::StructuredSDFGBuilder builder("sdfg_a");

    auto& struct_def_A = builder.add_structure("MyStructA", false);
    struct_def_A.add_member(types::Scalar(types::PrimitiveType::UInt8));

    auto& struct_def_B = builder.add_structure("MyStructB", false);
    struct_def_B.add_member(types::Structure("MyStructA"));

    auto sdfg = builder.move();

    ConditionalSchedule schedule(sdfg);

    codegen::CCodeGenerator generator(schedule, false);
    EXPECT_TRUE(generator.generate());

    auto result = generator.classes().str();
    EXPECT_EQ(result, R"(typedef struct MyStructB MyStructB;
typedef struct MyStructA MyStructA;
typedef struct MyStructA
{
char member_0;
} MyStructA;
typedef struct MyStructB
{
MyStructA member_0;
} MyStructB;
)");
}

TEST(CCodeGeneratorTest, DispatchGlobals) {
    builder::StructuredSDFGBuilder builder("sdfg_a");

    builder.add_container("a", types::Scalar(types::PrimitiveType::Int32), false, true);

    auto sdfg = builder.move();

    ConditionalSchedule schedule(sdfg);

    codegen::CCodeGenerator generator(schedule, false);
    EXPECT_TRUE(generator.generate());

    auto result = generator.globals().str();
    EXPECT_EQ(result, "extern int a;\n");
}
