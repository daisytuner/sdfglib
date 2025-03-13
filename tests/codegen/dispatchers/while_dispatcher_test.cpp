#include "sdfg/codegen/dispatchers/while_dispatcher.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/codegen/language_extensions/c_language_extension.h"
#include "sdfg/conditional_schedule.h"

using namespace sdfg;

TEST(WhileDispatcherTest, DispatchNode) {
    builder::StructuredSDFGBuilder builder("sdfg_a");
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto& loop = builder.add_while(root);

    auto final_sdfg = builder.move();

    ConditionalSchedule schedule(final_sdfg);

    codegen::CLanguageExtension language_extension;
    codegen::WhileDispatcher dispatcher(language_extension, schedule.schedule(0), loop, false);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::PrettyPrinter library_stream;
    dispatcher.dispatch_node(main_stream, globals_stream, library_stream);

    EXPECT_EQ(globals_stream.str(), "");
    EXPECT_EQ(library_stream.str(), "");
    EXPECT_EQ(main_stream.str(), "while (1)\n{\n}\n");
}

TEST(BreakDispatcherTest, DispatchNode) {
    builder::StructuredSDFGBuilder builder("sdfg_a");
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto& loop = builder.add_while(root);
    auto& break_node = builder.add_break(loop.root(), loop);

    auto final_sdfg = builder.move();

    ConditionalSchedule schedule(final_sdfg);

    codegen::CLanguageExtension language_extension;
    codegen::BreakDispatcher dispatcher(language_extension, schedule.schedule(0), break_node,
                                        false);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::PrettyPrinter library_stream;
    dispatcher.dispatch_node(main_stream, globals_stream, library_stream);

    EXPECT_EQ(globals_stream.str(), "");
    EXPECT_EQ(library_stream.str(), "");
    EXPECT_EQ(main_stream.str(), "break;\n");
}

TEST(ContinueDispatcherTest, DispatchNode) {
    builder::StructuredSDFGBuilder builder("sdfg_a");
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto& loop = builder.add_while(root);
    auto& continue_node = builder.add_continue(loop.root(), loop);

    auto final_sdfg = builder.move();

    ConditionalSchedule schedule(final_sdfg);

    codegen::CLanguageExtension language_extension;
    codegen::ContinueDispatcher dispatcher(language_extension, schedule.schedule(0), continue_node,
                                           false);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::PrettyPrinter library_stream;
    dispatcher.dispatch_node(main_stream, globals_stream, library_stream);

    EXPECT_EQ(globals_stream.str(), "");
    EXPECT_EQ(library_stream.str(), "");
    EXPECT_EQ(main_stream.str(), "continue;\n");
}

TEST(ReturnDispatcherTest, DispatchNode) {
    builder::StructuredSDFGBuilder builder("sdfg_a");
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto& return_node = builder.add_return(root);

    auto final_sdfg = builder.move();

    ConditionalSchedule schedule(final_sdfg);

    codegen::CLanguageExtension language_extension;
    codegen::ReturnDispatcher dispatcher(language_extension, schedule.schedule(0), return_node,
                                         false);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::PrettyPrinter library_stream;
    dispatcher.dispatch_node(main_stream, globals_stream, library_stream);

    EXPECT_EQ(globals_stream.str(), "");
    EXPECT_EQ(library_stream.str(), "");
    EXPECT_EQ(main_stream.str(), "return;\n");
}
