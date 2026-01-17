#include "sdfg/targets/highway/codegen/highway_map_dispatcher.h"

#include "sdfg/targets/highway/schedule.h"

#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/codegen/language_extensions/c_language_extension.h>
#include <sdfg/data_flow/library_nodes/math/math.h>
#include <sdfg/data_flow/tasklet.h>

#include <gtest/gtest.h>

using namespace sdfg;

TEST(HighwayMapDispatcherTest, TaskletOperations) {
    struct TestCase {
        data_flow::TaskletCode code;
        std::vector<std::string> inputs;
        std::string expected;
    };

    std::vector<TestCase> test_cases = {
        {data_flow::TaskletCode::assign, {"in1"}, "out = in1;"},
        {data_flow::TaskletCode::int_abs, {"in1"}, "out = hn::Abs(in1);"},
        {data_flow::TaskletCode::int_add, {"in1", "in2"}, "out = hn::Add(in1, in2);"},
        {data_flow::TaskletCode::int_sub, {"in1", "in2"}, "out = hn::Sub(in1, in2);"},
        {data_flow::TaskletCode::int_mul, {"in1", "in2"}, "out = hn::Mul(in1, in2);"},
        {data_flow::TaskletCode::int_sdiv, {"in1", "in2"}, "out = hn::Div(in1, in2);"},
        {data_flow::TaskletCode::int_and, {"in1", "in2"}, "out = hn::And(in1, in2);"},
        {data_flow::TaskletCode::int_or, {"in1", "in2"}, "out = hn::Or(in1, in2);"},
        {data_flow::TaskletCode::int_xor, {"in1", "in2"}, "out = hn::Xor(in1, in2);"},
        {data_flow::TaskletCode::int_smin, {"in1", "in2"}, "out = hn::Min(in1, in2);"},
        {data_flow::TaskletCode::int_smax, {"in1", "in2"}, "out = hn::Max(in1, in2);"},
        {data_flow::TaskletCode::int_umin, {"in1", "in2"}, "out = hn::Min(in1, in2);"},
        {data_flow::TaskletCode::int_umax, {"in1", "in2"}, "out = hn::Max(in1, in2);"},
        {data_flow::TaskletCode::int_eq, {"in1", "in2"}, "out = hn::Eq(in1, in2);"},
        {data_flow::TaskletCode::int_ne, {"in1", "in2"}, "out = hn::Ne(in1, in2);"},
        {data_flow::TaskletCode::int_sge, {"in1", "in2"}, "out = hn::Ge(in1, in2);"},
        {data_flow::TaskletCode::int_sgt, {"in1", "in2"}, "out = hn::Gt(in1, in2);"},
        {data_flow::TaskletCode::int_sle, {"in1", "in2"}, "out = hn::Le(in1, in2);"},
        {data_flow::TaskletCode::int_slt, {"in1", "in2"}, "out = hn::Lt(in1, in2);"},
        {data_flow::TaskletCode::int_uge, {"in1", "in2"}, "out = hn::Ge(in1, in2);"},
        {data_flow::TaskletCode::int_ugt, {"in1", "in2"}, "out = hn::Gt(in1, in2);"},
        {data_flow::TaskletCode::int_ule, {"in1", "in2"}, "out = hn::Le(in1, in2);"},
        {data_flow::TaskletCode::int_ult, {"in1", "in2"}, "out = hn::Lt(in1, in2);"},

        // FP ops
        {data_flow::TaskletCode::fp_neg, {"in1"}, "out = hn::Neg(in1);"},
        {data_flow::TaskletCode::fp_add, {"in1", "in2"}, "out = hn::Add(in1, in2);"},
        {data_flow::TaskletCode::fp_sub, {"in1", "in2"}, "out = hn::Sub(in1, in2);"},
        {data_flow::TaskletCode::fp_mul, {"in1", "in2"}, "out = hn::Mul(in1, in2);"},
        {data_flow::TaskletCode::fp_div, {"in1", "in2"}, "out = hn::Div(in1, in2);"},
        {data_flow::TaskletCode::fp_oeq, {"in1", "in2"}, "out = hn::Eq(in1, in2);"},
        {data_flow::TaskletCode::fp_one, {"in1", "in2"}, "out = hn::Ne(in1, in2);"},
        {data_flow::TaskletCode::fp_oge, {"in1", "in2"}, "out = hn::Ge(in1, in2);"},
        {data_flow::TaskletCode::fp_ogt, {"in1", "in2"}, "out = hn::Gt(in1, in2);"},
        {data_flow::TaskletCode::fp_ole, {"in1", "in2"}, "out = hn::Le(in1, in2);"},
        {data_flow::TaskletCode::fp_olt, {"in1", "in2"}, "out = hn::Lt(in1, in2);"},
        {data_flow::TaskletCode::fp_ueq, {"in1", "in2"}, "out = hn::Eq(in1, in2);"},
        {data_flow::TaskletCode::fp_une, {"in1", "in2"}, "out = hn::Ne(in1, in2);"},
        {data_flow::TaskletCode::fp_ugt, {"in1", "in2"}, "out = hn::Gt(in1, in2);"},
        {data_flow::TaskletCode::fp_uge, {"in1", "in2"}, "out = hn::Ge(in1, in2);"},
        {data_flow::TaskletCode::fp_ult, {"in1", "in2"}, "out = hn::Lt(in1, in2);"},
        {data_flow::TaskletCode::fp_ule, {"in1", "in2"}, "out = hn::Le(in1, in2);"},
        {data_flow::TaskletCode::fp_fma, {"in1", "in2", "in3"}, "out = hn::MulAdd(in1, in2, in3);"},
    };

    for (const auto& test_case : test_cases) {
        builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
        structured_control_flow::Block& block = builder.add_block(builder.subject().root());

        auto& tasklet = builder.add_tasklet(block, test_case.code, "out", test_case.inputs);
        std::string result = highway::HighwayMapDispatcher::tasklet(tasklet);
        EXPECT_EQ(result, test_case.expected);
    }
}

TEST(HighwayMapDispatcherTest, TaskletUnsupported) {
    std::vector<data_flow::TaskletCode> unsupported = {
        data_flow::TaskletCode::int_srem,
        data_flow::TaskletCode::int_udiv,
        data_flow::TaskletCode::int_urem,
        data_flow::TaskletCode::int_shl,
        data_flow::TaskletCode::int_ashr,
        data_flow::TaskletCode::int_lshr,
        data_flow::TaskletCode::int_scmp,
        data_flow::TaskletCode::int_ucmp,
        data_flow::TaskletCode::fp_rem,
        data_flow::TaskletCode::fp_ord,
        data_flow::TaskletCode::fp_uno
    };

    for (auto code : unsupported) {
        builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
        structured_control_flow::Block& block = builder.add_block(builder.subject().root());

        auto& tasklet = builder.add_tasklet(block, code, "out", {"in1", "in2"});
        EXPECT_THROW(highway::HighwayMapDispatcher::tasklet(tasklet), std::runtime_error);
    }
}

TEST(HighwayMapDispatcherTest, CMathFunctions) {
    struct TestCase {
        math::cmath::CMathFunction func;
        std::string expected;
    };

    std::vector<TestCase> test_cases = {
        {math::cmath::CMathFunction::cos, "_out = hn::Cos(_in1);"},
        {math::cmath::CMathFunction::ceil, "_out = hn::Ceil(_in1);"},
        {math::cmath::CMathFunction::exp, "_out = hn::Exp(_in1);"},
        {math::cmath::CMathFunction::exp2, "_out = hn::Exp2(_in1);"},
        {math::cmath::CMathFunction::fabs, "_out = hn::Abs(_in1);"},
        {math::cmath::CMathFunction::floor, "_out = hn::Floor(_in1);"},
        {math::cmath::CMathFunction::log, "_out = hn::Log(_in1);"},
        {math::cmath::CMathFunction::log2, "_out = hn::Log2(_in1);"},
        {math::cmath::CMathFunction::log10, "_out = hn::Log10(_in1);"},
        {math::cmath::CMathFunction::sin, "_out = hn::Sin(_in1);"},
        {math::cmath::CMathFunction::sqrt, "_out = hn::Sqrt(_in1);"},
        {math::cmath::CMathFunction::trunc, "_out = hn::Trunc(_in1);"},
        {math::cmath::CMathFunction::round, "_out = hn::Round(_in1);"},
    };

    for (const auto& test_case : test_cases) {
        builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
        structured_control_flow::Block& block = builder.add_block(builder.subject().root());

        auto& node = builder.add_library_node<
            math::cmath::CMathNode>(block, DebugInfo(), test_case.func, types::PrimitiveType::Float);
        std::string result = highway::HighwayMapDispatcher::cmath_node(static_cast<math::cmath::CMathNode&>(node));
        EXPECT_EQ(result, test_case.expected);
    }
}

TEST(HighwayMapDispatcherTest, DaisyVec) {
    // Create a dummy map to satisfy constructor
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    auto& map = builder.add_map(
        builder.subject().root(),
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::zero(),
        symbolic::add(symbolic::symbol("i"), symbolic::one()),
        highway::ScheduleType_Highway::create()
    );

    analysis::AnalysisManager analysis_manager(builder.subject());
    codegen::CLanguageExtension language_extension(builder.subject());
    auto instrumentation = codegen::InstrumentationPlan::none(builder.subject());
    auto arg_capture = codegen::ArgCapturePlan::none(builder.subject());

    highway::HighwayMapDispatcher
        dispatcher(language_extension, builder.subject(), analysis_manager, map, *instrumentation, *arg_capture);

    EXPECT_EQ(dispatcher.daisy_vec(types::PrimitiveType::Int8), "daisy_vec_s8");
    EXPECT_EQ(dispatcher.daisy_vec(types::PrimitiveType::Int16), "daisy_vec_s16");
    EXPECT_EQ(dispatcher.daisy_vec(types::PrimitiveType::Int32), "daisy_vec_s32");
    EXPECT_EQ(dispatcher.daisy_vec(types::PrimitiveType::Int64), "daisy_vec_s64");
    EXPECT_EQ(dispatcher.daisy_vec(types::PrimitiveType::UInt8), "daisy_vec_u8");
    EXPECT_EQ(dispatcher.daisy_vec(types::PrimitiveType::UInt16), "daisy_vec_u16");
    EXPECT_EQ(dispatcher.daisy_vec(types::PrimitiveType::UInt32), "daisy_vec_u32");
    EXPECT_EQ(dispatcher.daisy_vec(types::PrimitiveType::UInt64), "daisy_vec_u64");
    EXPECT_EQ(dispatcher.daisy_vec(types::PrimitiveType::Float), "daisy_vec_f32");
    EXPECT_EQ(dispatcher.daisy_vec(types::PrimitiveType::Double), "daisy_vec_f64");
}
