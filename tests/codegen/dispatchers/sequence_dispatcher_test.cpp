#include "sdfg/codegen/dispatchers/sequence_dispatcher.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/codegen/language_extensions/c_language_extension.h"
#include "sdfg/conditional_schedule.h"

using namespace sdfg;

TEST(SequenceDispatcherTest, DispatchNode_Empty) {
    builder::StructuredSDFGBuilder builder("sdfg_a");
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto final_sdfg = builder.move();

    ConditionalSchedule schedule(final_sdfg);

    codegen::CLanguageExtension language_extension;
    codegen::SequenceDispatcher dispatcher(language_extension, schedule.schedule(0), root, false);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::PrettyPrinter library_stream;
    dispatcher.dispatch_node(main_stream, globals_stream, library_stream);

    EXPECT_EQ(globals_stream.str(), "");
    EXPECT_EQ(library_stream.str(), "");
    EXPECT_EQ(main_stream.str(), "");
}

TEST(SequenceDispatcherTest, DispatchNode_Transition) {
    builder::StructuredSDFGBuilder builder("sdfg_a");
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    builder.add_container("i", types::Scalar(types::PrimitiveType::Int16));
    auto& block1 =
        builder.add_block(root, {{symbolic::symbol("i"), symbolic::integer(0)}}, DebugInfo());

    auto final_sdfg = builder.move();

    ConditionalSchedule schedule(final_sdfg);

    codegen::CLanguageExtension language_extension;
    codegen::SequenceDispatcher dispatcher(language_extension, schedule.schedule(0), root, false);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::PrettyPrinter library_stream;
    dispatcher.dispatch_node(main_stream, globals_stream, library_stream);

    EXPECT_EQ(globals_stream.str(), "");
    EXPECT_EQ(library_stream.str(), "");
    EXPECT_EQ(main_stream.str(), "{\n    i = 0;\n}\n");
}

TEST(SequenceDispatcherTest, DispatchNode_MultipleBlocks) {
    builder::StructuredSDFGBuilder builder("sdfg_a");
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    builder.add_container("i", types::Scalar(types::PrimitiveType::Int16));
    auto& block1 =
        builder.add_block(root, {{symbolic::symbol("i"), symbolic::integer(0)}}, DebugInfo());
    auto& block2 =
        builder.add_block(root, {{symbolic::symbol("i"), symbolic::integer(1)}}, DebugInfo());

    auto final_sdfg = builder.move();

    ConditionalSchedule schedule(final_sdfg);

    codegen::CLanguageExtension language_extension;
    codegen::SequenceDispatcher dispatcher(language_extension, schedule.schedule(0), root, false);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::PrettyPrinter library_stream;
    dispatcher.dispatch_node(main_stream, globals_stream, library_stream);

    EXPECT_EQ(globals_stream.str(), "");
    EXPECT_EQ(library_stream.str(), "");
    EXPECT_EQ(main_stream.str(), "{\n    i = 0;\n}\n{\n    i = 1;\n}\n");
}
