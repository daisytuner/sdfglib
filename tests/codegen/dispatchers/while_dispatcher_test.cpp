#include "sdfg/codegen/dispatchers/while_dispatcher.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/codegen/language_extensions/c_language_extension.h"

using namespace sdfg;

TEST(WhileDispatcherTest, DispatchNode) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto& loop = builder.add_while(root);

    auto final_sdfg = builder.move();

    codegen::CLanguageExtension language_extension;
    codegen::Instrumentation instrumentation(*final_sdfg);
    codegen::WhileDispatcher dispatcher(language_extension, *final_sdfg, loop, instrumentation);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::PrettyPrinter library_stream;
    dispatcher.dispatch_node(main_stream, globals_stream, library_stream);

    EXPECT_EQ(globals_stream.str(), "");
    EXPECT_EQ(library_stream.str(), "");
    EXPECT_EQ(main_stream.str(), "while (1)\n{\n}\n");
}

TEST(BreakDispatcherTest, DispatchNode) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto& loop = builder.add_while(root);
    auto& break_node = builder.add_break(loop.root());

    auto final_sdfg = builder.move();

    codegen::CLanguageExtension language_extension;
    codegen::Instrumentation instrumentation(*final_sdfg);
    codegen::BreakDispatcher dispatcher(language_extension, *final_sdfg, break_node,
                                        instrumentation);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::PrettyPrinter library_stream;
    dispatcher.dispatch_node(main_stream, globals_stream, library_stream);

    EXPECT_EQ(globals_stream.str(), "");
    EXPECT_EQ(library_stream.str(), "");
    EXPECT_EQ(main_stream.str(), "break;\n");
}

TEST(ContinueDispatcherTest, DispatchNode) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto& loop = builder.add_while(root);
    auto& continue_node = builder.add_continue(loop.root());

    auto final_sdfg = builder.move();

    codegen::CLanguageExtension language_extension;
    codegen::Instrumentation instrumentation(*final_sdfg);
    codegen::ContinueDispatcher dispatcher(language_extension, *final_sdfg, continue_node,
                                           instrumentation);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::PrettyPrinter library_stream;
    dispatcher.dispatch_node(main_stream, globals_stream, library_stream);

    EXPECT_EQ(globals_stream.str(), "");
    EXPECT_EQ(library_stream.str(), "");
    EXPECT_EQ(main_stream.str(), "continue;\n");
}

TEST(ReturnDispatcherTest, DispatchNode) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto& return_node = builder.add_return(root);

    auto final_sdfg = builder.move();

    codegen::CLanguageExtension language_extension;
    codegen::Instrumentation instrumentation(*final_sdfg);
    codegen::ReturnDispatcher dispatcher(language_extension, *final_sdfg, return_node,
                                         instrumentation);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::PrettyPrinter library_stream;
    dispatcher.dispatch_node(main_stream, globals_stream, library_stream);

    EXPECT_EQ(globals_stream.str(), "");
    EXPECT_EQ(library_stream.str(), "");
    EXPECT_EQ(main_stream.str(), "return;\n");
}
