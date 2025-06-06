#include "sdfg/codegen/dispatchers/schedules/highway_dispatcher.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/codegen/language_extensions/c_language_extension.h"
#include "sdfg/conditional_schedule.h"

using namespace sdfg;

TEST(HighwayDispatcherTest, DispatchNode_Empty) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    builder.add_container("N", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    auto& loop = builder.add_for(
        root, symbolic::symbol("i"), symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0), symbolic::add(symbolic::symbol("i"), symbolic::integer(1)));

    auto final_sdfg = builder.move();

    ConditionalSchedule schedule(final_sdfg);

    codegen::CLanguageExtension language_extension;
    codegen::HighwayDispatcher dispatcher(language_extension, schedule.schedule(0), loop, false);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::PrettyPrinter library_stream;
    dispatcher.dispatch(main_stream, globals_stream, library_stream);

    EXPECT_EQ(globals_stream.str(), "extern void __node_1_lib(int *N, int *i);\n");
    EXPECT_EQ(main_stream.str(), "__node_1_lib(&N, &i);\n");
    EXPECT_EQ(library_stream.str(), R"(namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

HWY_ATTR void __node_1(int &N, int &i)
{
    const hn::ScalableTag<double> daisy_vec_fp;
    const hn::ScalableTag<int64_t> daisy_vec_i;
    const hn::ScalableTag<uint64_t> daisy_vec_ui;
    i = 0;
    {
        for(;(i + hn::Lanes(daisy_vec_i)) < N;i = i + hn::Lanes(daisy_vec_i))
        {
            {
            }
        }
    }
    {
        for(;i < N;i = i + 1)
        {
        }
    }
}

}
#if HWY_ONCE

HWY_EXPORT(__node_1);
extern "C" void __node_1_lib(int *N, int *i)
{
    HWY_STATIC_DISPATCH(__node_1)(*N, *i);
}

#endif
)");
}

TEST(HighwayDispatcherTest, DispatchNode_64) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    builder.add_container("N", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container(
        "a", types::Array(types::Scalar(types::PrimitiveType::Double), symbolic::symbol("N")));
    builder.add_container(
        "b", types::Array(types::Scalar(types::PrimitiveType::Double), symbolic::symbol("N")));

    auto& loop = builder.add_for(
        root, symbolic::symbol("i"), symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0), symbolic::add(symbolic::symbol("i"), symbolic::integer(1)));
    auto& block = builder.add_block(loop.root());
    auto& a = builder.add_access(block, "a");
    auto& b = builder.add_access(block, "b");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                        {"_out", types::Scalar(types::PrimitiveType::Double)},
                                        {{"_in", types::Scalar(types::PrimitiveType::Double)}});
    builder.add_memlet(block, a, "void", tasklet, "_in", data_flow::Subset{symbolic::symbol("i")});
    builder.add_memlet(block, tasklet, "_out", b, "void", data_flow::Subset{symbolic::symbol("i")});

    auto final_sdfg = builder.move();

    ConditionalSchedule schedule(final_sdfg);

    codegen::CLanguageExtension language_extension;
    codegen::HighwayDispatcher dispatcher(language_extension, schedule.schedule(0), loop, false);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::PrettyPrinter library_stream;
    dispatcher.dispatch(main_stream, globals_stream, library_stream);

    EXPECT_EQ(globals_stream.str(),
              "extern void __node_1_lib(int *N, double *a, double *b, int *i);\n");
    EXPECT_EQ(main_stream.str(), "__node_1_lib(&N, a, b, &i);\n");
    EXPECT_EQ(library_stream.str(), R"(namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

HWY_ATTR void __node_1(int &N, double *a, double *b, int &i)
{
    const hn::ScalableTag<double> daisy_vec_fp;
    const hn::ScalableTag<int64_t> daisy_vec_i;
    const hn::ScalableTag<uint64_t> daisy_vec_ui;
    i = 0;
    {
        for(;(i + hn::Lanes(daisy_vec_i)) < N;i = i + hn::Lanes(daisy_vec_i))
        {
            {
                {
                    {
                                                const auto _in = hn::LoadU(daisy_vec_fp, a + (i));
                        
                        const auto _out = (_in);
                        
                        hn::StoreU(_out, daisy_vec_fp, b + (i));
}
                }
            }
        }
    }
    {
        for(;i < N;i = i + 1)
        {
                {
                    double _in = a[i];
                    double _out;

                    _out = _in;

                    b[i] = _out;
                }
        }
    }
}

}
#if HWY_ONCE

HWY_EXPORT(__node_1);
extern "C" void __node_1_lib(int *N, double *a, double *b, int *i)
{
    HWY_STATIC_DISPATCH(__node_1)(*N, a, b, *i);
}

#endif
)");
}

TEST(HighwayDispatcherTest, DispatchNode_32) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    builder.add_container("N", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container(
        "a", types::Array(types::Scalar(types::PrimitiveType::Float), symbolic::symbol("N")));
    builder.add_container(
        "b", types::Array(types::Scalar(types::PrimitiveType::Float), symbolic::symbol("N")));

    auto& loop = builder.add_for(
        root, symbolic::symbol("i"), symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0), symbolic::add(symbolic::symbol("i"), symbolic::integer(1)));
    auto& block = builder.add_block(loop.root());
    auto& a = builder.add_access(block, "a");
    auto& b = builder.add_access(block, "b");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                        {"_out", types::Scalar(types::PrimitiveType::Float)},
                                        {{"_in", types::Scalar(types::PrimitiveType::Float)}});
    builder.add_memlet(block, a, "void", tasklet, "_in", data_flow::Subset{symbolic::symbol("i")});
    builder.add_memlet(block, tasklet, "_out", b, "void", data_flow::Subset{symbolic::symbol("i")});

    auto final_sdfg = builder.move();

    ConditionalSchedule schedule(final_sdfg);

    codegen::CLanguageExtension language_extension;
    codegen::HighwayDispatcher dispatcher(language_extension, schedule.schedule(0), loop, false);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::PrettyPrinter library_stream;
    dispatcher.dispatch(main_stream, globals_stream, library_stream);

    EXPECT_EQ(globals_stream.str(),
              "extern void __node_1_lib(int *N, float *a, float *b, int *i);\n");
    EXPECT_EQ(main_stream.str(), "__node_1_lib(&N, a, b, &i);\n");
    EXPECT_EQ(library_stream.str(), R"(namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

HWY_ATTR void __node_1(int &N, float *a, float *b, int &i)
{
    const hn::ScalableTag<float> daisy_vec_fp;
    const hn::ScalableTag<int32_t> daisy_vec_i;
    const hn::ScalableTag<uint32_t> daisy_vec_ui;
    i = 0;
    {
        for(;(i + hn::Lanes(daisy_vec_i)) < N;i = i + hn::Lanes(daisy_vec_i))
        {
            {
                {
                    {
                                                const auto _in = hn::LoadU(daisy_vec_fp, a + (i));
                        
                        const auto _out = (_in);
                        
                        hn::StoreU(_out, daisy_vec_fp, b + (i));
}
                }
            }
        }
    }
    {
        for(;i < N;i = i + 1)
        {
                {
                    float _in = a[i];
                    float _out;

                    _out = _in;

                    b[i] = _out;
                }
        }
    }
}

}
#if HWY_ONCE

HWY_EXPORT(__node_1);
extern "C" void __node_1_lib(int *N, float *a, float *b, int *i)
{
    HWY_STATIC_DISPATCH(__node_1)(*N, a, b, *i);
}

#endif
)");
}

TEST(HighwayDispatcherTest, DispatchNode_16) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    builder.add_container("N", types::Scalar(types::PrimitiveType::Int16));
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int16));
    builder.add_container(
        "a", types::Array(types::Scalar(types::PrimitiveType::Int16), symbolic::symbol("N")));
    builder.add_container(
        "b", types::Array(types::Scalar(types::PrimitiveType::Int16), symbolic::symbol("N")));

    auto& loop = builder.add_for(
        root, symbolic::symbol("i"), symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0), symbolic::add(symbolic::symbol("i"), symbolic::integer(1)));
    auto& block = builder.add_block(loop.root());
    auto& a = builder.add_access(block, "a");
    auto& b = builder.add_access(block, "b");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                        {"_out", types::Scalar(types::PrimitiveType::Int16)},
                                        {{"_in", types::Scalar(types::PrimitiveType::Int16)}});
    builder.add_memlet(block, a, "void", tasklet, "_in", data_flow::Subset{symbolic::symbol("i")});
    builder.add_memlet(block, tasklet, "_out", b, "void", data_flow::Subset{symbolic::symbol("i")});

    auto final_sdfg = builder.move();

    ConditionalSchedule schedule(final_sdfg);

    codegen::CLanguageExtension language_extension;
    codegen::HighwayDispatcher dispatcher(language_extension, schedule.schedule(0), loop, false);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::PrettyPrinter library_stream;
    dispatcher.dispatch(main_stream, globals_stream, library_stream);

    EXPECT_EQ(globals_stream.str(),
              "extern void __node_1_lib(short *N, short *a, short *b, short *i);\n");
    EXPECT_EQ(main_stream.str(), "__node_1_lib(&N, a, b, &i);\n");
    EXPECT_EQ(library_stream.str(), R"(namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

HWY_ATTR void __node_1(short &N, short *a, short *b, short &i)
{
    const hn::ScalableTag<int16_t> daisy_vec_i;
    const hn::ScalableTag<uint16_t> daisy_vec_ui;
    i = 0;
    {
        for(;(i + hn::Lanes(daisy_vec_i)) < N;i = i + hn::Lanes(daisy_vec_i))
        {
            {
                {
                    {
                                                const auto _in = hn::LoadU(daisy_vec_i, a + (i));
                        
                        const auto _out = (_in);
                        
                        hn::StoreU(_out, daisy_vec_i, b + (i));
}
                }
            }
        }
    }
    {
        for(;i < N;i = i + 1)
        {
                {
                    short _in = a[i];
                    short _out;

                    _out = _in;

                    b[i] = _out;
                }
        }
    }
}

}
#if HWY_ONCE

HWY_EXPORT(__node_1);
extern "C" void __node_1_lib(short *N, short *a, short *b, short *i)
{
    HWY_STATIC_DISPATCH(__node_1)(*N, a, b, *i);
}

#endif
)");
}

TEST(HighwayDispatcherTest, DispatchNode_8) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    builder.add_container("N", types::Scalar(types::PrimitiveType::Int8));
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int8));
    builder.add_container(
        "a", types::Array(types::Scalar(types::PrimitiveType::Int8), symbolic::symbol("N")));
    builder.add_container(
        "b", types::Array(types::Scalar(types::PrimitiveType::Int8), symbolic::symbol("N")));

    auto& loop = builder.add_for(
        root, symbolic::symbol("i"), symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0), symbolic::add(symbolic::symbol("i"), symbolic::integer(1)));
    auto& block = builder.add_block(loop.root());
    auto& a = builder.add_access(block, "a");
    auto& b = builder.add_access(block, "b");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                        {"_out", types::Scalar(types::PrimitiveType::Int8)},
                                        {{"_in", types::Scalar(types::PrimitiveType::Int8)}});
    builder.add_memlet(block, a, "void", tasklet, "_in", data_flow::Subset{symbolic::symbol("i")});
    builder.add_memlet(block, tasklet, "_out", b, "void", data_flow::Subset{symbolic::symbol("i")});

    auto final_sdfg = builder.move();

    ConditionalSchedule schedule(final_sdfg);

    codegen::CLanguageExtension language_extension;
    codegen::HighwayDispatcher dispatcher(language_extension, schedule.schedule(0), loop, false);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::PrettyPrinter library_stream;
    dispatcher.dispatch(main_stream, globals_stream, library_stream);

    EXPECT_EQ(globals_stream.str(),
              "extern void __node_1_lib(signed char *N, signed char *a, signed char *b, signed "
              "char *i);\n");
    EXPECT_EQ(main_stream.str(), "__node_1_lib(&N, a, b, &i);\n");
    EXPECT_EQ(library_stream.str(), R"(namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

HWY_ATTR void __node_1(signed char &N, signed char *a, signed char *b, signed char &i)
{
    const hn::ScalableTag<int8_t> daisy_vec_i;
    const hn::ScalableTag<uint8_t> daisy_vec_ui;
    i = 0;
    {
        for(;(i + hn::Lanes(daisy_vec_i)) < N;i = i + hn::Lanes(daisy_vec_i))
        {
            {
                {
                    {
                                                const auto _in = hn::LoadU(daisy_vec_i, a + (i));
                        
                        const auto _out = (_in);
                        
                        hn::StoreU(_out, daisy_vec_i, b + (i));
}
                }
            }
        }
    }
    {
        for(;i < N;i = i + 1)
        {
                {
                    signed char _in = a[i];
                    signed char _out;

                    _out = _in;

                    b[i] = _out;
                }
        }
    }
}

}
#if HWY_ONCE

HWY_EXPORT(__node_1);
extern "C" void __node_1_lib(signed char *N, signed char *a, signed char *b, signed char *i)
{
    HWY_STATIC_DISPATCH(__node_1)(*N, a, b, *i);
}

#endif
)");
}

TEST(HighwayDispatcherTest, TaskletToSIMDInstruction_Assign) {
    data_flow::TaskletCode c = data_flow::TaskletCode::assign;
    EXPECT_EQ(codegen::HighwayDispatcher::tasklet_to_simd_instruction(c, "daisy_vec_fp"), "(");
}

TEST(HighwayDispatcherTest, TaskletToSIMDInstruction_Neg) {
    data_flow::TaskletCode c = data_flow::TaskletCode::neg;
    EXPECT_EQ(codegen::HighwayDispatcher::tasklet_to_simd_instruction(c, "daisy_vec_fp"),
              "hn::Neg(");
}

TEST(HighwayDispatcherTest, TaskletToSIMDInstruction_Add) {
    data_flow::TaskletCode c = data_flow::TaskletCode::add;
    EXPECT_EQ(codegen::HighwayDispatcher::tasklet_to_simd_instruction(c, "daisy_vec_fp"),
              "hn::Add(");
}

TEST(HighwayDispatcherTest, TaskletToSIMDInstruction_Sub) {
    data_flow::TaskletCode c = data_flow::TaskletCode::sub;
    EXPECT_EQ(codegen::HighwayDispatcher::tasklet_to_simd_instruction(c, "daisy_vec_fp"),
              "hn::Sub(");
}

TEST(HighwayDispatcherTest, TaskletToSIMDInstruction_Mul) {
    data_flow::TaskletCode c = data_flow::TaskletCode::mul;
    EXPECT_EQ(codegen::HighwayDispatcher::tasklet_to_simd_instruction(c, "daisy_vec_fp"),
              "hn::Mul(");
}

TEST(HighwayDispatcherTest, TaskletToSIMDInstruction_Div) {
    data_flow::TaskletCode c = data_flow::TaskletCode::div;
    EXPECT_EQ(codegen::HighwayDispatcher::tasklet_to_simd_instruction(c, "daisy_vec_fp"),
              "hn::Div(");
}

TEST(HighwayDispatcherTest, TaskletToSIMDInstruction_Mod) {
    data_flow::TaskletCode c = data_flow::TaskletCode::mod;
    EXPECT_EQ(codegen::HighwayDispatcher::tasklet_to_simd_instruction(c, "daisy_vec_fp"),
              "hn::Mod(");
}

TEST(HighwayDispatcherTest, TaskletToSIMDInstruction_Abs) {
    data_flow::TaskletCode c = data_flow::TaskletCode::abs;
    EXPECT_EQ(codegen::HighwayDispatcher::tasklet_to_simd_instruction(c, "daisy_vec_fp"),
              "hn::Abs(");
}

TEST(HighwayDispatcherTest, TaskletToSIMDInstruction_Max) {
    data_flow::TaskletCode c = data_flow::TaskletCode::max;
    EXPECT_EQ(codegen::HighwayDispatcher::tasklet_to_simd_instruction(c, "daisy_vec_fp"),
              "hn::Max(");
}

TEST(HighwayDispatcherTest, TaskletToSIMDInstruction_Min) {
    data_flow::TaskletCode c = data_flow::TaskletCode::min;
    EXPECT_EQ(codegen::HighwayDispatcher::tasklet_to_simd_instruction(c, "daisy_vec_fp"),
              "hn::Min(");
}

TEST(HighwayDispatcherTest, TaskletToSIMDInstruction_FAbs) {
    data_flow::TaskletCode c = data_flow::TaskletCode::fabs;
    EXPECT_EQ(codegen::HighwayDispatcher::tasklet_to_simd_instruction(c, "daisy_vec_fp"),
              "hn::Abs(");
}

TEST(HighwayDispatcherTest, TaskletToSIMDInstruction_Sqrt) {
    data_flow::TaskletCode c = data_flow::TaskletCode::sqrt;
    EXPECT_EQ(codegen::HighwayDispatcher::tasklet_to_simd_instruction(c, "daisy_vec_fp"),
              "hn::Sqrt(");
}

TEST(HighwayDispatcherTest, TaskletToSIMDInstruction_FSqrt) {
    data_flow::TaskletCode c = data_flow::TaskletCode::sqrtf;
    EXPECT_EQ(codegen::HighwayDispatcher::tasklet_to_simd_instruction(c, "daisy_vec_fp"),
              "hn::Sqrt(");
}

TEST(HighwayDispatcherTest, TaskletToSIMDInstruction_Sin) {
    data_flow::TaskletCode c = data_flow::TaskletCode::sin;
    EXPECT_EQ(codegen::HighwayDispatcher::tasklet_to_simd_instruction(c, "daisy_vec_fp"),
              "hn::Sin(daisy_vec_fp, ");
}

TEST(HighwayDispatcherTest, TaskletToSIMDInstruction_Cos) {
    data_flow::TaskletCode c = data_flow::TaskletCode::cos;
    EXPECT_EQ(codegen::HighwayDispatcher::tasklet_to_simd_instruction(c, "daisy_vec_fp"),
              "hn::Cos(daisy_vec_fp, ");
}

TEST(HighwayDispatcherTest, TaskletToSIMDInstruction_Tan) {
    data_flow::TaskletCode c = data_flow::TaskletCode::tan;
    EXPECT_EQ(codegen::HighwayDispatcher::tasklet_to_simd_instruction(c, "daisy_vec_fp"),
              "hn::Tan(daisy_vec_fp, ");
}

TEST(HighwayDispatcherTest, TaskletToSIMDInstruction_Pow) {
    data_flow::TaskletCode c = data_flow::TaskletCode::pow;
    EXPECT_EQ(codegen::HighwayDispatcher::tasklet_to_simd_instruction(c, "daisy_vec_fp"),
              "hn::Pow(daisy_vec_fp, ");
}

TEST(HighwayDispatcherTest, TaskletToSIMDInstruction_Exp) {
    data_flow::TaskletCode c = data_flow::TaskletCode::exp;
    EXPECT_EQ(codegen::HighwayDispatcher::tasklet_to_simd_instruction(c, "daisy_vec_fp"),
              "hn::Exp(daisy_vec_fp, ");
}

TEST(HighwayDispatcherTest, TaskletToSIMDInstruction_Expf) {
    data_flow::TaskletCode c = data_flow::TaskletCode::expf;
    EXPECT_EQ(codegen::HighwayDispatcher::tasklet_to_simd_instruction(c, "daisy_vec_fp"),
              "hn::Exp(daisy_vec_fp, ");
}

TEST(HighwayDispatcherTest, TaskletToSIMDInstruction_Exp2) {
    data_flow::TaskletCode c = data_flow::TaskletCode::exp2;
    EXPECT_EQ(codegen::HighwayDispatcher::tasklet_to_simd_instruction(c, "daisy_vec_fp"),
              "hn::Exp2(daisy_vec_fp, ");
}

TEST(HighwayDispatcherTest, TaskletToSIMDInstruction_Log) {
    data_flow::TaskletCode c = data_flow::TaskletCode::log;
    EXPECT_EQ(codegen::HighwayDispatcher::tasklet_to_simd_instruction(c, "daisy_vec_fp"),
              "hn::Log(daisy_vec_fp, ");
}

TEST(HighwayDispatcherTest, TaskletToSIMDInstruction_Log2) {
    data_flow::TaskletCode c = data_flow::TaskletCode::log2;
    EXPECT_EQ(codegen::HighwayDispatcher::tasklet_to_simd_instruction(c, "daisy_vec_fp"),
              "hn::Log2(daisy_vec_fp, ");
}

TEST(HighwayDispatcherTest, TaskletToSIMDInstruction_Log10) {
    data_flow::TaskletCode c = data_flow::TaskletCode::log10;
    EXPECT_EQ(codegen::HighwayDispatcher::tasklet_to_simd_instruction(c, "daisy_vec_fp"),
              "hn::Log10(daisy_vec_fp, ");
}

TEST(HighwayDispatcherTest, TaskletToSIMDInstruction_FMA) {
    data_flow::TaskletCode c = data_flow::TaskletCode::fma;
    EXPECT_EQ(codegen::HighwayDispatcher::tasklet_to_simd_instruction(c, "daisy_vec_fp"),
              "hn::MulAdd(");
}

TEST(HighwayDispatcherTest, TaskletToSIMDInstruction_Floor) {
    data_flow::TaskletCode c = data_flow::TaskletCode::floor;
    EXPECT_EQ(codegen::HighwayDispatcher::tasklet_to_simd_instruction(c, "daisy_vec_fp"),
              "hn::Floor(daisy_vec_fp, ");
}

TEST(HighwayDispatcherTest, TaskletToSIMDInstruction_Ceil) {
    data_flow::TaskletCode c = data_flow::TaskletCode::ceil;
    EXPECT_EQ(codegen::HighwayDispatcher::tasklet_to_simd_instruction(c, "daisy_vec_fp"),
              "hn::Ceil(daisy_vec_fp, ");
}

TEST(HighwayDispatcherTest, TaskletToSIMDInstruction_Trunc) {
    data_flow::TaskletCode c = data_flow::TaskletCode::trunc;
    EXPECT_EQ(codegen::HighwayDispatcher::tasklet_to_simd_instruction(c, "daisy_vec_fp"),
              "hn::Trunc(daisy_vec_fp, ");
}

TEST(HighwayDispatcherTest, TaskletToSIMDInstruction_Round) {
    data_flow::TaskletCode c = data_flow::TaskletCode::round;
    EXPECT_EQ(codegen::HighwayDispatcher::tasklet_to_simd_instruction(c, "daisy_vec_fp"),
              "hn::Round(daisy_vec_fp, ");
}

TEST(HighwayDispatcherTest, TaskletToSIMDInstruction_Lt) {
    data_flow::TaskletCode c = data_flow::TaskletCode::olt;
    EXPECT_EQ(codegen::HighwayDispatcher::tasklet_to_simd_instruction(c, "daisy_vec_fp"),
              "hn::Lt(");
}

TEST(HighwayDispatcherTest, TaskletToSIMDInstruction_Le) {
    data_flow::TaskletCode c = data_flow::TaskletCode::ole;
    EXPECT_EQ(codegen::HighwayDispatcher::tasklet_to_simd_instruction(c, "daisy_vec_fp"),
              "hn::Le(");
}

TEST(HighwayDispatcherTest, TaskletToSIMDInstruction_Eq) {
    data_flow::TaskletCode c = data_flow::TaskletCode::oeq;
    EXPECT_EQ(codegen::HighwayDispatcher::tasklet_to_simd_instruction(c, "daisy_vec_fp"),
              "hn::Eq(");
}

TEST(HighwayDispatcherTest, TaskletToSIMDInstruction_Ne) {
    data_flow::TaskletCode c = data_flow::TaskletCode::one;
    EXPECT_EQ(codegen::HighwayDispatcher::tasklet_to_simd_instruction(c, "daisy_vec_fp"),
              "hn::Ne(");
}

TEST(HighwayDispatcherTest, TaskletToSIMDInstruction_Gt) {
    data_flow::TaskletCode c = data_flow::TaskletCode::ogt;
    EXPECT_EQ(codegen::HighwayDispatcher::tasklet_to_simd_instruction(c, "daisy_vec_fp"),
              "hn::Gt(");
}

TEST(HighwayDispatcherTest, TaskletToSIMDInstruction_Ge) {
    data_flow::TaskletCode c = data_flow::TaskletCode::oge;
    EXPECT_EQ(codegen::HighwayDispatcher::tasklet_to_simd_instruction(c, "daisy_vec_fp"),
              "hn::Ge(");
}

TEST(HighwayDispatcherTest, TaskletToSIMDInstruction_Bitwise_and) {
    data_flow::TaskletCode c = data_flow::TaskletCode::bitwise_and;
    EXPECT_EQ(codegen::HighwayDispatcher::tasklet_to_simd_instruction(c, "daisy_vec_fp"),
              "hn::And(");
}

TEST(HighwayDispatcherTest, TaskletToSIMDInstruction_Bitwise_or) {
    data_flow::TaskletCode c = data_flow::TaskletCode::bitwise_or;
    EXPECT_EQ(codegen::HighwayDispatcher::tasklet_to_simd_instruction(c, "daisy_vec_fp"),
              "hn::Or(");
}

TEST(HighwayDispatcherTest, TaskletToSIMDInstruction_Bitwise_xor) {
    data_flow::TaskletCode c = data_flow::TaskletCode::bitwise_xor;
    EXPECT_EQ(codegen::HighwayDispatcher::tasklet_to_simd_instruction(c, "daisy_vec_fp"),
              "hn::Xor(");
}

TEST(HighwayDispatcherTest, TaskletToSIMDInstruction_Bitwise_not) {
    data_flow::TaskletCode c = data_flow::TaskletCode::bitwise_not;
    EXPECT_EQ(codegen::HighwayDispatcher::tasklet_to_simd_instruction(c, "daisy_vec_fp"),
              "hn::Not(");
}
