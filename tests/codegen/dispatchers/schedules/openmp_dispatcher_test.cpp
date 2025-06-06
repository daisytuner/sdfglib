#include "sdfg/codegen/dispatchers/schedules/openmp_dispatcher.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/codegen/language_extensions/c_language_extension.h"
#include "sdfg/conditional_schedule.h"

using namespace sdfg;

TEST(OpenMPDispatcherTest, DispatchNode) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType::CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    auto& loop = builder.add_for(
        root, symbolic::symbol("i"), symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0), symbolic::add(symbolic::symbol("i"), symbolic::integer(1)));

    auto final_sdfg = builder.move();

    ConditionalSchedule schedule(final_sdfg);

    codegen::CLanguageExtension language_extension;
    codegen::OpenMPDispatcher dispatcher(language_extension, schedule.schedule(0), loop, false);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::PrettyPrinter library_stream;
    dispatcher.dispatch_node(main_stream, globals_stream, library_stream);

    EXPECT_EQ(globals_stream.str(), "");
    EXPECT_EQ(library_stream.str(), "");
    EXPECT_EQ(main_stream.str(),
              "#pragma omp parallel for private(i)\nfor(i = 0;i < 10;i = 1 + i)\n{\n}\n");
}
