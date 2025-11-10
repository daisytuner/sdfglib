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
    analysis::AnalysisManager analysis_manager(*final_sdfg);

    codegen::CLanguageExtension language_extension;
    auto instrumentation = codegen::InstrumentationPlan::none(*final_sdfg);
    auto arg_capture = codegen::ArgCapturePlan::none(*final_sdfg);
    codegen::WhileDispatcher
        dispatcher(language_extension, *final_sdfg, analysis_manager, loop, *instrumentation, *arg_capture);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::CodeSnippetFactory library_factory;
    dispatcher.dispatch_node(main_stream, globals_stream, library_factory);

    EXPECT_EQ(globals_stream.str(), "");
    EXPECT_TRUE(library_factory.snippets().empty());
    EXPECT_EQ(main_stream.str(), "while (1)\n{\n}\n");
}

TEST(BreakDispatcherTest, DispatchNode) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto& loop = builder.add_while(root);
    auto& break_node = builder.add_break(loop.root());

    auto final_sdfg = builder.move();
    analysis::AnalysisManager analysis_manager(*final_sdfg);

    codegen::CLanguageExtension language_extension;
    auto instrumentation = codegen::InstrumentationPlan::none(*final_sdfg);
    auto arg_capture = codegen::ArgCapturePlan::none(*final_sdfg);
    codegen::BreakDispatcher
        dispatcher(language_extension, *final_sdfg, analysis_manager, break_node, *instrumentation, *arg_capture);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::CodeSnippetFactory library_factory;
    dispatcher.dispatch_node(main_stream, globals_stream, library_factory);

    EXPECT_EQ(globals_stream.str(), "");
    EXPECT_TRUE(library_factory.snippets().empty());
    EXPECT_EQ(main_stream.str(), "break;\n");
}

TEST(ContinueDispatcherTest, DispatchNode) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto& loop = builder.add_while(root);
    auto& continue_node = builder.add_continue(loop.root());

    auto final_sdfg = builder.move();
    analysis::AnalysisManager analysis_manager(*final_sdfg);

    codegen::CLanguageExtension language_extension;
    auto instrumentation = codegen::InstrumentationPlan::none(*final_sdfg);
    auto arg_capture = codegen::ArgCapturePlan::none(*final_sdfg);
    codegen::ContinueDispatcher
        dispatcher(language_extension, *final_sdfg, analysis_manager, continue_node, *instrumentation, *arg_capture);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::CodeSnippetFactory library_factory;
    dispatcher.dispatch_node(main_stream, globals_stream, library_factory);

    EXPECT_EQ(globals_stream.str(), "");
    EXPECT_TRUE(library_factory.snippets().empty());
    EXPECT_EQ(main_stream.str(), "continue;\n");
}

TEST(ReturnDispatcherTest, DispatchNode) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto& return_node = builder.add_return(root, "");

    auto final_sdfg = builder.move();
    analysis::AnalysisManager analysis_manager(*final_sdfg);

    codegen::CLanguageExtension language_extension;
    auto instrumentation = codegen::InstrumentationPlan::none(*final_sdfg);
    auto arg_capture = codegen::ArgCapturePlan::none(*final_sdfg);
    codegen::ReturnDispatcher
        dispatcher(language_extension, *final_sdfg, analysis_manager, return_node, *instrumentation, *arg_capture);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::CodeSnippetFactory library_factory;
    dispatcher.dispatch_node(main_stream, globals_stream, library_factory);

    EXPECT_EQ(globals_stream.str(), "");
    EXPECT_TRUE(library_factory.snippets().empty());
    EXPECT_EQ(main_stream.str(), "return ;\n");
}
