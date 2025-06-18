#include "sdfg/codegen/dispatchers/map_dispatcher.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/codegen/language_extensions/c_language_extension.h"

using namespace sdfg;

TEST(MapDispatcherTest, DispatchNode) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    auto& loop = builder.add_map(root, symbolic::symbol("i"), symbolic::integer(10),
                                 structured_control_flow::ScheduleType_Sequential);

    auto final_sdfg = builder.move();

    codegen::CLanguageExtension language_extension;
    codegen::Instrumentation instrumentation(*final_sdfg);
    codegen::MapDispatcher dispatcher(language_extension, *final_sdfg, loop, instrumentation);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::PrettyPrinter library_stream;
    dispatcher.dispatch_node(main_stream, globals_stream, library_stream);

    EXPECT_EQ(globals_stream.str(), "");
    EXPECT_EQ(library_stream.str(), "");
    EXPECT_EQ(main_stream.str(), "// Map\nfor(i = 0; i < 10; i++)\n{\n}\n");
}
