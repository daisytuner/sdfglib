#include "sdfg/codegen/dispatchers/node_dispatcher_factory.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/codegen/language_extensions/c_language_extension.h"
#include "sdfg/conditional_schedule.h"

using namespace sdfg;

TEST(NodeDispatcherFactoryTest, CreateDispatch_Block) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType::CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto& block = builder.add_block(root);

    auto final_sdfg = builder.move();

    ConditionalSchedule schedule(final_sdfg);

    codegen::CLanguageExtension language_extension;
    codegen::Instrumentation instrumentation(schedule.schedule(0));
    auto dispatcher = codegen::create_dispatcher(language_extension, schedule.schedule(0), block,
                                                 instrumentation);
    EXPECT_TRUE(dynamic_cast<codegen::BlockDispatcher*>(dispatcher.get()));
}

TEST(NodeDispatcherFactoryTest, CreateDispatch_Sequence) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType::CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto& sequence = builder.add_sequence(root);

    auto final_sdfg = builder.move();

    ConditionalSchedule schedule(final_sdfg);

    codegen::CLanguageExtension language_extension;
    codegen::Instrumentation instrumentation(schedule.schedule(0));
    auto dispatcher = codegen::create_dispatcher(language_extension, schedule.schedule(0), sequence,
                                                 instrumentation);
    EXPECT_TRUE(dynamic_cast<codegen::SequenceDispatcher*>(dispatcher.get()));
}

TEST(NodeDispatcherFactoryTest, CreateDispatch_IfElse) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType::CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto& if_else = builder.add_if_else(root);

    auto final_sdfg = builder.move();

    ConditionalSchedule schedule(final_sdfg);

    codegen::CLanguageExtension language_extension;
    codegen::Instrumentation instrumentation(schedule.schedule(0));
    auto dispatcher = codegen::create_dispatcher(language_extension, schedule.schedule(0), if_else,
                                                 instrumentation);
    EXPECT_TRUE(dynamic_cast<codegen::IfElseDispatcher*>(dispatcher.get()));
}

TEST(NodeDispatcherFactoryTest, CreateDispatch_While) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType::CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto& while_loop = builder.add_while(root);

    auto final_sdfg = builder.move();

    ConditionalSchedule schedule(final_sdfg);

    codegen::CLanguageExtension language_extension;
    codegen::Instrumentation instrumentation(schedule.schedule(0));
    auto dispatcher = codegen::create_dispatcher(language_extension, schedule.schedule(0),
                                                 while_loop, instrumentation);
    EXPECT_TRUE(dynamic_cast<codegen::WhileDispatcher*>(dispatcher.get()));
}

TEST(NodeDispatcherFactoryTest, CreateDispatch_For) {
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
    codegen::Instrumentation instrumentation(schedule.schedule(0));
    auto dispatcher =
        codegen::create_dispatcher(language_extension, schedule.schedule(0), loop, instrumentation);
    EXPECT_TRUE(dynamic_cast<codegen::ForDispatcher*>(dispatcher.get()));
}

TEST(NodeDispatcherFactoryTest, CreateDispatch_Return) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType::CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto& return_node = builder.add_return(root);

    auto final_sdfg = builder.move();

    ConditionalSchedule schedule(final_sdfg);

    codegen::CLanguageExtension language_extension;
    codegen::Instrumentation instrumentation(schedule.schedule(0));
    auto dispatcher = codegen::create_dispatcher(language_extension, schedule.schedule(0),
                                                 return_node, instrumentation);
    EXPECT_TRUE(dynamic_cast<codegen::ReturnDispatcher*>(dispatcher.get()));
}

TEST(NodeDispatcherFactoryTest, CreateDispatch_Break) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType::CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto& while_loop = builder.add_while(root);
    auto& break_node = builder.add_break(while_loop.root());

    auto final_sdfg = builder.move();

    ConditionalSchedule schedule(final_sdfg);

    codegen::CLanguageExtension language_extension;
    codegen::Instrumentation instrumentation(schedule.schedule(0));
    auto dispatcher = codegen::create_dispatcher(language_extension, schedule.schedule(0),
                                                 break_node, instrumentation);
    EXPECT_TRUE(dynamic_cast<codegen::BreakDispatcher*>(dispatcher.get()));
}

TEST(NodeDispatcherFactoryTest, CreateDispatch_Continue) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType::CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto& while_loop = builder.add_while(root);
    auto& continue_node = builder.add_continue(while_loop.root());

    auto final_sdfg = builder.move();

    ConditionalSchedule schedule(final_sdfg);

    codegen::CLanguageExtension language_extension;
    codegen::Instrumentation instrumentation(schedule.schedule(0));
    auto dispatcher = codegen::create_dispatcher(language_extension, schedule.schedule(0),
                                                 continue_node, instrumentation);
    EXPECT_TRUE(dynamic_cast<codegen::ContinueDispatcher*>(dispatcher.get()));
}
