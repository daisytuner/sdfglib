#include "sdfg/codegen/dispatchers/map_dispatcher.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/codegen/language_extensions/c_language_extension.h"
#include "sdfg/structured_control_flow/map.h"

using namespace sdfg;

TEST(MapDispatcherTest, DispatchNode) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    auto& loop = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    auto final_sdfg = builder.move();
    analysis::AnalysisManager analysis_manager(*final_sdfg);

    codegen::CLanguageExtension language_extension;
    auto instrumentation = codegen::InstrumentationPlan::none(*final_sdfg);
    auto arg_capture = codegen::ArgCapturePlan::none(*final_sdfg);
    codegen::MapDispatcher
        dispatcher(language_extension, *final_sdfg, analysis_manager, loop, *instrumentation, *arg_capture);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::CodeSnippetFactory library_factory;
    dispatcher.dispatch_node(main_stream, globals_stream, library_factory);

    EXPECT_EQ(globals_stream.str(), "");
    EXPECT_TRUE(library_factory.snippets().empty());
    EXPECT_EQ(main_stream.str(), "// Map\nfor(i = 0;i < 10;i = 1 + i)\n{\n}\n");
}

TEST(CPU_PARALLELMapDispatcherTest, DispatchNode) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    auto& loop = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_CPU_Parallel::create()
    );

    auto final_sdfg = builder.move();
    analysis::AnalysisManager analysis_manager(*final_sdfg);

    codegen::CLanguageExtension language_extension;
    auto instrumentation = codegen::InstrumentationPlan::none(*final_sdfg);
    auto arg_capture = codegen::ArgCapturePlan::none(*final_sdfg);
    codegen::CPUParallelMapDispatcher
        dispatcher(language_extension, *final_sdfg, analysis_manager, loop, *instrumentation, *arg_capture);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::CodeSnippetFactory library_factory;
    dispatcher.dispatch_node(main_stream, globals_stream, library_factory);

    EXPECT_EQ(globals_stream.str(), "");
    EXPECT_TRUE(library_factory.snippets().empty());
    EXPECT_EQ(main_stream.str(), "// Map\n#pragma omp parallel for schedule(static)\nfor(i = 0;i < 10;i = 1 + i)\n{\n}\n");
}

TEST(CPU_PARALLELMapDispatcherTest, DispatchNodeScheduleDynamic) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    auto& loop = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_CPU_Parallel::create()
    );

    ScheduleType schedule = structured_control_flow::ScheduleType_CPU_Parallel::create();

    ScheduleType_CPU_Parallel::omp_schedule(schedule, OpenMPSchedule::Dynamic);

    builder.update_schedule_type(loop, schedule);

    auto final_sdfg = builder.move();
    analysis::AnalysisManager analysis_manager(*final_sdfg);

    codegen::CLanguageExtension language_extension;
    auto instrumentation = codegen::InstrumentationPlan::none(*final_sdfg);
    auto arg_capture = codegen::ArgCapturePlan::none(*final_sdfg);
    codegen::CPUParallelMapDispatcher
        dispatcher(language_extension, *final_sdfg, analysis_manager, loop, *instrumentation, *arg_capture);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::CodeSnippetFactory library_factory;
    dispatcher.dispatch_node(main_stream, globals_stream, library_factory);

    EXPECT_EQ(globals_stream.str(), "");
    EXPECT_TRUE(library_factory.snippets().empty());
    EXPECT_EQ(
        main_stream.str(), "// Map\n#pragma omp parallel for schedule(dynamic)\nfor(i = 0;i < 10;i = 1 + i)\n{\n}\n"
    );
}

TEST(CPU_PARALLELMapDispatcherTest, DispatchNodeNumThreads) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    auto& loop = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_CPU_Parallel::create()
    );

    ScheduleType schedule = structured_control_flow::ScheduleType_CPU_Parallel::create();

    ScheduleType_CPU_Parallel::omp_schedule(schedule, OpenMPSchedule::Dynamic);
    ScheduleType_CPU_Parallel::num_threads(schedule, symbolic::integer(4));

    builder.update_schedule_type(loop, schedule);

    auto final_sdfg = builder.move();
    analysis::AnalysisManager analysis_manager(*final_sdfg);

    codegen::CLanguageExtension language_extension;
    auto instrumentation = codegen::InstrumentationPlan::none(*final_sdfg);
    auto arg_capture = codegen::ArgCapturePlan::none(*final_sdfg);
    codegen::CPUParallelMapDispatcher
        dispatcher(language_extension, *final_sdfg, analysis_manager, loop, *instrumentation, *arg_capture);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::CodeSnippetFactory library_factory;
    dispatcher.dispatch_node(main_stream, globals_stream, library_factory);

    EXPECT_EQ(globals_stream.str(), "");
    EXPECT_TRUE(library_factory.snippets().empty());
    EXPECT_EQ(
        main_stream.str(),
        "// Map\n#pragma omp parallel for schedule(dynamic) num_threads(4)\nfor(i = 0;i < 10;i = 1 + i)\n{\n}\n"
    );
}
