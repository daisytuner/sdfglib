#include "sdfg/codegen/dispatchers/sequence_dispatcher.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/codegen/language_extensions/c_language_extension.h"
#include "sdfg/debug_info.h"

using namespace sdfg;

TEST(SequenceDispatcherTest, DispatchNode_Empty) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU, DebugTable());
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto final_sdfg = builder.move();

    codegen::CLanguageExtension language_extension;
    auto instrumentation = codegen::InstrumentationPlan::none(*final_sdfg);
    codegen::SequenceDispatcher dispatcher(language_extension, *final_sdfg, root, *instrumentation);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::CodeSnippetFactory library_factory;
    dispatcher.dispatch_node(main_stream, globals_stream, library_factory);

    EXPECT_EQ(globals_stream.str(), "");
    EXPECT_TRUE(library_factory.snippets().empty());
    EXPECT_EQ(main_stream.str(), "");
}

TEST(SequenceDispatcherTest, DispatchNode_Transition) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU, DebugTable());
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    builder.add_container("i", types::Scalar(types::PrimitiveType::Int16));
    auto& block1 = builder.add_block(root, {{symbolic::symbol("i"), symbolic::integer(0)}});

    auto final_sdfg = builder.move();

    codegen::CLanguageExtension language_extension;
    auto instrumentation = codegen::InstrumentationPlan::none(*final_sdfg);
    codegen::SequenceDispatcher dispatcher(language_extension, *final_sdfg, root, *instrumentation);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::CodeSnippetFactory library_factory;
    dispatcher.dispatch_node(main_stream, globals_stream, library_factory);

    EXPECT_EQ(globals_stream.str(), "");
    EXPECT_TRUE(library_factory.snippets().empty());
    EXPECT_EQ(main_stream.str(), "{\n    i = 0;\n}\n");
}

TEST(SequenceDispatcherTest, DispatchNode_MultipleBlocks) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU, DebugTable());
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    builder.add_container("i", types::Scalar(types::PrimitiveType::Int16));
    auto& block1 = builder.add_block(root, {{symbolic::symbol("i"), symbolic::integer(0)}});
    auto& block2 = builder.add_block(root, {{symbolic::symbol("i"), symbolic::integer(1)}});

    auto final_sdfg = builder.move();

    codegen::CLanguageExtension language_extension;
    auto instrumentation = codegen::InstrumentationPlan::none(*final_sdfg);
    codegen::SequenceDispatcher dispatcher(language_extension, *final_sdfg, root, *instrumentation);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::CodeSnippetFactory library_factory;
    dispatcher.dispatch_node(main_stream, globals_stream, library_factory);

    EXPECT_EQ(globals_stream.str(), "");
    EXPECT_TRUE(library_factory.snippets().empty());
    EXPECT_EQ(main_stream.str(), "{\n    i = 0;\n}\n{\n    i = 1;\n}\n");
}
