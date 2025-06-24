#include "sdfg/codegen/code_generators/c_code_generator.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/codegen/instrumentation/instrumentation_strategy.h"
#include "sdfg/codegen/instrumentation/capture_var_plan.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/type.h"

using namespace sdfg;

TEST(CCodeGeneratorTest, FunctionDefintion) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto sdfg = builder.move();

    codegen::CCodeGenerator generator(*sdfg);
    auto result = generator.function_definition();
    EXPECT_EQ(result, "extern void sdfg_a()");
}

TEST(CCodeGeneratorTest, Dispatch_Includes) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto sdfg = builder.move();

    codegen::CCodeGenerator generator(*sdfg);
    EXPECT_TRUE(generator.generate());

    auto result = generator.includes().str();
    EXPECT_EQ(result,
              "#include <math.h>\n#include <stdbool.h>\n#include <stdlib.h>\n#define "
              "__daisy_min(a,b) ((a)<(b)?(a):(b))\n#define __daisy_max(a,b) "
              "((a)>(b)?(a):(b))\n#define __daisy_fma(a,b,c) a * b + c\n");
}

TEST(CCodeGeneratorTest, DispatchStructures_Basic) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);

    auto& struct_def_A = builder.add_structure("MyStructA", false);
    struct_def_A.add_member(types::Scalar(types::PrimitiveType::UInt8));

    auto sdfg = builder.move();

    codegen::CCodeGenerator generator(*sdfg);
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
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);

    auto& struct_def_A = builder.add_structure("MyStructA", false);
    struct_def_A.add_member(types::Scalar(types::PrimitiveType::UInt8));

    auto& struct_def_B = builder.add_structure("MyStructB", false);
    struct_def_B.add_member(types::Structure("MyStructA"));

    auto sdfg = builder.move();

    codegen::CCodeGenerator generator(*sdfg);
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
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);

    builder.add_container("a", types::Scalar(types::PrimitiveType::Int32), false, true);

    auto sdfg = builder.move();

    codegen::CCodeGenerator generator(*sdfg);
    EXPECT_TRUE(generator.generate());

    auto result = generator.globals().str();
    EXPECT_EQ(result, "extern int a;\n");
}

TEST(CCodeGeneratorTest, CaptureInstrumentationInit) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);

    auto sdfg = builder.move();

    codegen::CCodeGenerator generator(*sdfg, codegen::InstrumentationStrategy::NONE, true);
    std::stringstream output;

    generator.emit_capture_context_init(output);

    EXPECT_EQ(output.str(), R"(static void* __capture_ctx;
static void __attribute__((constructor(1000))) __capture_ctx_init(void) {
	__capture_ctx = __daisy_capture_init("sdfg_a");
}

)");
}

TEST(CCodeGeneratorTest, EmitArgInCaptures) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    builder.add_container("arg0", types::Scalar(types::PrimitiveType::Int64), true, false);
    auto innerType = types::Array(types::Scalar(types::PrimitiveType::Float), symbolic::integer(210));
    builder.add_container("arg1", types::Pointer(static_cast<types::IType&>(innerType)), true, false);
    builder.add_container("ext0", types::Scalar(types::PrimitiveType::Int64), false, true);

    auto sdfg = builder.move();

    codegen::CCodeGenerator generator(*sdfg, codegen::InstrumentationStrategy::NONE, true);
    std::stringstream output;

    std::vector<codegen::CaptureVarPlan> plan = {
        {true, false, codegen::CaptureVarType::CapRaw, 0, false, types::PrimitiveType::Int64},
        {true, false, codegen::CaptureVarType::Cap2D, 1, false, types::PrimitiveType::Float, symbolic::integer(190), symbolic::integer(210)},
        {true, false, codegen::CaptureVarType::CapRaw, 2, true, types::PrimitiveType::Int64}
    };

    generator.emit_arg_captures(output, plan, false);

    EXPECT_EQ(
        output.str(),
        R"(const bool __daisy_cap_en = __daisy_capture_enter(__capture_ctx);
if (__daisy_cap_en) {
	__daisy_capture_raw(__capture_ctx, 0, &arg0, sizeof(arg0), 5, false);
	__daisy_capture_2d(__capture_ctx, 1, arg1, sizeof(float), 14, 190, 210, false);
	__daisy_capture_raw(__capture_ctx, 2, &ext0, sizeof(ext0), 5, false);
}
)");
}

TEST(CCodeGeneratorTest, EmitArgOutCaptures) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    builder.add_container("arg0", types::Scalar(types::PrimitiveType::Int64), true, false);
    auto innerType = types::Array(types::Scalar(types::PrimitiveType::Float), symbolic::integer(210));
    builder.add_container("arg1", types::Pointer(static_cast<types::IType&>(innerType)), true, false);
    builder.add_container("ext0", types::Scalar(types::PrimitiveType::Int64), false, true);
    auto pType = types::Scalar(types::PrimitiveType::Int64);
    builder.add_container("ext1", types::Pointer(static_cast<types::IType&>(pType)), false, true);

    auto sdfg = builder.move();

    codegen::CCodeGenerator generator(*sdfg, codegen::InstrumentationStrategy::NONE, true);
    std::stringstream output;

    std::vector<codegen::CaptureVarPlan> plan = {
        {true, false, codegen::CaptureVarType::CapRaw, 0, false, types::PrimitiveType::Int64},
        {true, true, codegen::CaptureVarType::Cap2D, 1, false, types::PrimitiveType::Float, symbolic::integer(190), symbolic::integer(210)},
        {true, false, codegen::CaptureVarType::CapRaw, 2, true, types::PrimitiveType::Int64},
        {false, true, codegen::CaptureVarType::Cap1D, 3, true, types::PrimitiveType::Int64, symbolic::integer(1)},
    };

    generator.emit_arg_captures(output, plan, true);

    EXPECT_EQ(
        output.str(),
        R"(if (__daisy_cap_en) {
	__daisy_capture_2d(__capture_ctx, 1, arg1, sizeof(float), 14, 190, 210, true);
	__daisy_capture_1d(__capture_ctx, 3, ext1, sizeof(long long), 5, 1, true);
	__daisy_capture_end(__capture_ctx);
}
)");
}

/**
 * This actually tests only code in CodeGenerator. But its abstract, so just reuse the generator, without actually using any of its code.
 */
TEST(CCodeGeneratorTest, CreateCapturePlans) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    builder.add_container("arg0", types::Scalar(types::PrimitiveType::Int64), true, false);
    auto valueType = types::Scalar(types::PrimitiveType::Float);
    auto innerType = types::Array(valueType, symbolic::integer(210));
    builder.add_container("arg1", types::Pointer(static_cast<types::IType&>(innerType)), true, false);
    builder.add_container("ext0", types::Scalar(types::PrimitiveType::Int64), false, true);
    builder.add_container("ext1", types::Pointer(static_cast<types::IType&>(valueType)), false, true);

    auto sym_i = symbolic::symbol("i");
    auto sym_j = symbolic::symbol("j");

    auto& root = builder.subject().root();
    auto& outer_for = builder.add_for(root, sym_i, symbolic::Lt(sym_i, symbolic::integer(190)), symbolic::zero(), symbolic::add(sym_i, symbolic::one()));
    auto& inner_for = builder.add_for(outer_for.root(), sym_j, symbolic::Lt(sym_j, symbolic::integer(210)), symbolic::zero(), symbolic::add(sym_j, symbolic::one()));

    auto& block = builder.add_block(inner_for.root());
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::add, {"__out", valueType},
        {
            {"__in0", valueType},
            {"1", valueType}
        });
    auto& readArr = builder.add_access(block, "arg1");
    auto& readPrev = builder.add_memlet(block, readArr, "void", tasklet, "__in0", {sym_i, sym_j});
    auto& writeArr = builder.add_access(block, "arg1");
    auto& writeEntry = builder.add_memlet(block, tasklet, "__out", writeArr, "void", {sym_i, sym_j});
    auto& writeOut = builder.add_access(block, "ext1");
    auto& writeLast = builder.add_memlet(block, tasklet, "__out", writeOut, "void", {symbolic::zero()});

    auto sdfg = builder.move();

    codegen::CCodeGenerator generator(*sdfg, codegen::InstrumentationStrategy::NONE, true);
    auto capturePlan = generator.create_capture_plans();

    EXPECT_EQ(capturePlan->size(), 4);

    EXPECT_EQ((*capturePlan)[0].capture_input, true);
    EXPECT_EQ((*capturePlan)[0].capture_output, false);
    EXPECT_EQ((*capturePlan)[0].type, codegen::CaptureVarType::CapRaw);
    EXPECT_EQ((*capturePlan)[0].arg_idx, 0);
    EXPECT_EQ((*capturePlan)[0].is_external, false);
    EXPECT_EQ((*capturePlan)[0].inner_type, types::PrimitiveType::Int64);

    EXPECT_EQ((*capturePlan)[1].capture_input, true);
    EXPECT_EQ((*capturePlan)[1].capture_output, true);
    EXPECT_EQ((*capturePlan)[1].type, codegen::CaptureVarType::Cap2D);
    EXPECT_EQ((*capturePlan)[1].arg_idx, 1);
    EXPECT_EQ((*capturePlan)[1].is_external, false);
    EXPECT_EQ((*capturePlan)[1].inner_type, types::PrimitiveType::Float);
    EXPECT_TRUE(symbolic::eq((*capturePlan)[1].dim1, symbolic::integer(190))) << "dim1: " << (*capturePlan)[1].dim1->__str__() << std::endl;;
    EXPECT_TRUE(symbolic::eq((*capturePlan)[1].dim2, symbolic::integer(210))) << "dim2: " << (*capturePlan)[1].dim2->__str__() << std::endl;

    EXPECT_EQ((*capturePlan)[2].capture_input, true);
    EXPECT_EQ((*capturePlan)[2].capture_output, false);
    EXPECT_EQ((*capturePlan)[2].type, codegen::CaptureVarType::CapRaw);
    EXPECT_EQ((*capturePlan)[2].arg_idx, 2);
    EXPECT_EQ((*capturePlan)[2].is_external, true);

    EXPECT_EQ((*capturePlan)[3].capture_input, false);
    EXPECT_EQ((*capturePlan)[3].capture_output, true);
    EXPECT_EQ((*capturePlan)[3].type, codegen::CaptureVarType::Cap1D);
    EXPECT_EQ((*capturePlan)[3].arg_idx, 3);
    EXPECT_EQ((*capturePlan)[3].is_external, true);
    EXPECT_EQ((*capturePlan)[3].inner_type, types::PrimitiveType::Float);
    EXPECT_TRUE(symbolic::eq((*capturePlan)[3].dim1, symbolic::integer(1)));
}