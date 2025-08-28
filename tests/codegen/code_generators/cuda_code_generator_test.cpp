#include "sdfg/codegen/code_generators/cuda_code_generator.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"

using namespace sdfg;

TEST(CUDACodeGeneratorTest, FunctionDefintion) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_NV_GLOBAL, DebugInfo());
    auto sdfg = builder.move();

    auto instrumentation_plan = codegen::InstrumentationPlan::none(*sdfg);
    codegen::CUDACodeGenerator generator(*sdfg, *instrumentation_plan);
    auto result = generator.function_definition();
    EXPECT_EQ(result, "extern \"C\" __global__ void sdfg_a()");
}

TEST(CUDACodeGeneratorTest, Dispatch_Includes) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_NV_GLOBAL, DebugInfo());
    auto sdfg = builder.move();

    auto instrumentation_plan = codegen::InstrumentationPlan::none(*sdfg);
    codegen::CUDACodeGenerator generator(*sdfg, *instrumentation_plan);
    EXPECT_TRUE(generator.generate());

    auto result = generator.includes().str();
    EXPECT_EQ(result, "#define __DAISY_NVVM__\n#include <daisy_rtl/daisy_rtl.h>\n");
}

TEST(CUDACodeGeneratorTest, DispatchStructures_Basic) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_NV_GLOBAL, DebugInfo());

    auto& struct_def_A = builder.add_structure("MyStructA", false);
    struct_def_A.add_member(types::Scalar(types::PrimitiveType::UInt8));

    auto sdfg = builder.move();

    auto instrumentation_plan = codegen::InstrumentationPlan::none(*sdfg);
    codegen::CUDACodeGenerator generator(*sdfg, *instrumentation_plan);
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
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_NV_GLOBAL, DebugInfo());

    auto& struct_def_A = builder.add_structure("MyStructA", false);
    struct_def_A.add_member(types::Scalar(types::PrimitiveType::UInt8));

    auto& struct_def_B = builder.add_structure("MyStructB", false);
    struct_def_B.add_member(types::Structure("MyStructA"));

    auto sdfg = builder.move();

    auto instrumentation_plan = codegen::InstrumentationPlan::none(*sdfg);
    codegen::CUDACodeGenerator generator(*sdfg, *instrumentation_plan);
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
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_NV_GLOBAL, DebugInfo());

    sdfg::types::Scalar base_type(sdfg::types::StorageType_NV_Global, 0, "", sdfg::types::PrimitiveType::Int32);
    sdfg::types::Pointer
        ptr_type(sdfg::types::StorageType_NV_Global, 0, "", static_cast<sdfg::types::IType&>(base_type));
    builder.add_container("a", ptr_type, false, true);

    auto sdfg = builder.move();

    auto instrumentation_plan = codegen::InstrumentationPlan::none(*sdfg);
    codegen::CUDACodeGenerator generator(*sdfg, *instrumentation_plan);
    EXPECT_TRUE(generator.generate());

    auto result = generator.globals().str();
    EXPECT_EQ(result, "extern int a;\n");
}
