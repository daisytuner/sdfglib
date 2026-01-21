#include "sdfg/targets/omp/codegen/omp_map_dispatcher.h"
#include "sdfg/targets/omp/schedule.h"

#include <sdfg/codegen/language_extensions/c_language_extension.h>

#include <gtest/gtest.h>

using namespace sdfg;

TEST(OMPMapDispatcherTest, DispatchNode) {
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
        omp::ScheduleType_OMP::create()
    );

    auto final_sdfg = builder.move();
    analysis::AnalysisManager analysis_manager(*final_sdfg);

    codegen::CLanguageExtension language_extension(*final_sdfg);
    auto instrumentation = codegen::InstrumentationPlan::none(*final_sdfg);
    auto arg_capture = codegen::ArgCapturePlan::none(*final_sdfg);
    omp::OMPMapDispatcher
        dispatcher(language_extension, *final_sdfg, analysis_manager, loop, *instrumentation, *arg_capture);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::CodeSnippetFactory library_factory;
    dispatcher.dispatch_node(main_stream, globals_stream, library_factory);

    EXPECT_EQ(globals_stream.str(), "");
    EXPECT_TRUE(library_factory.snippets().empty());
    EXPECT_EQ(main_stream.str(), "// Map\n#pragma omp parallel for schedule(static)\nfor(i = 0;i < 10;i = 1 + i)\n{\n}\n");
}

TEST(OMPMapDispatcherTest, DispatchNodeScheduleDynamic) {
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
        omp::ScheduleType_OMP::create()
    );

    ScheduleType schedule = omp::ScheduleType_OMP::create();

    omp::ScheduleType_OMP::omp_schedule(schedule, omp::OpenMPSchedule::Dynamic);

    builder.update_schedule_type(loop, schedule);

    auto final_sdfg = builder.move();
    analysis::AnalysisManager analysis_manager(*final_sdfg);

    codegen::CLanguageExtension language_extension(*final_sdfg);
    auto instrumentation = codegen::InstrumentationPlan::none(*final_sdfg);
    auto arg_capture = codegen::ArgCapturePlan::none(*final_sdfg);
    omp::OMPMapDispatcher
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

TEST(OMPMapDispatcherTest, DispatchNodeNumThreads) {
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
        omp::ScheduleType_OMP::create()
    );

    ScheduleType schedule = omp::ScheduleType_OMP::create();

    omp::ScheduleType_OMP::omp_schedule(schedule, omp::OpenMPSchedule::Dynamic);
    omp::ScheduleType_OMP::num_threads(schedule, symbolic::integer(4));

    builder.update_schedule_type(loop, schedule);

    auto final_sdfg = builder.move();
    analysis::AnalysisManager analysis_manager(*final_sdfg);

    codegen::CLanguageExtension language_extension(*final_sdfg);
    auto instrumentation = codegen::InstrumentationPlan::none(*final_sdfg);
    auto arg_capture = codegen::ArgCapturePlan::none(*final_sdfg);
    omp::OMPMapDispatcher
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
