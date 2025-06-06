#include "sdfg/codegen/code_generators/cuda_code_generator.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/conditional_schedule.h"

using namespace sdfg;

TEST(CUDACodeGeneratorTest, FunctionDefintion) {
    builder::StructuredSDFGBuilder builder("sdfg_a");
    auto sdfg = builder.move();

    ConditionalSchedule schedule(sdfg);

    codegen::CUDACodeGenerator generator(schedule);
    auto result = generator.function_definition();
    EXPECT_EQ(result, "extern \"C\" __global__ void sdfg_a()");
}

TEST(CUDACodeGeneratorTest, Dispatch_Includes) {
    builder::StructuredSDFGBuilder builder("sdfg_a");
    auto sdfg = builder.move();

    ConditionalSchedule schedule(sdfg);

    codegen::CUDACodeGenerator generator(schedule);
    EXPECT_TRUE(generator.generate());

    auto result = generator.includes().str();
    EXPECT_EQ(result,
              "#define __DAISY_NVVM__\n#include \"daisyrtl.h\"\n#define __daisy_min(a,b) "
              "((a)<(b)?(a):(b))\n#define __daisy_max(a,b) ((a)>(b)?(a):(b))\n#define "
              "__daisy_fma(a,b,c) a * b + c\n");
}

TEST(CUDACodeGeneratorTest, DispatchStructures_Basic) {
    builder::StructuredSDFGBuilder builder("sdfg_a");

    auto& struct_def_A = builder.add_structure("MyStructA", false);
    struct_def_A.add_member(types::Scalar(types::PrimitiveType::UInt8));

    auto sdfg = builder.move();

    ConditionalSchedule schedule(sdfg);

    codegen::CUDACodeGenerator generator(schedule);
    EXPECT_TRUE(generator.generate());

    auto result = generator.classes().str();
    EXPECT_EQ(result, R"(struct MyStructA;
struct MyStructA
{
char member_0;
};
)");
}

TEST(CUDACodeGeneratorTest, DispatchStructures_Nested) {
    builder::StructuredSDFGBuilder builder("sdfg_a");

    auto& struct_def_A = builder.add_structure("MyStructA", false);
    struct_def_A.add_member(types::Scalar(types::PrimitiveType::UInt8));

    auto& struct_def_B = builder.add_structure("MyStructB", false);
    struct_def_B.add_member(types::Structure("MyStructA"));

    auto sdfg = builder.move();

    ConditionalSchedule schedule(sdfg);

    codegen::CUDACodeGenerator generator(schedule);
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

TEST(CUDACodeGeneratorTest, DispatchGlobals) {
    builder::StructuredSDFGBuilder builder("sdfg_a");

    builder.add_container(
        "a", types::Scalar(types::StorageType::NV_Global, 0, "", types::PrimitiveType::Int32),
        false, true);

    auto sdfg = builder.move();

    ConditionalSchedule schedule(sdfg);

    codegen::CUDACodeGenerator generator(schedule);
    EXPECT_TRUE(generator.generate());

    auto result = generator.globals().str();
    EXPECT_EQ(result, "extern int a;\n");
}
