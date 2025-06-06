#include "sdfg/codegen/dispatchers/block_dispatcher.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/codegen/language_extensions/c_language_extension.h"
#include "sdfg/codegen/language_extensions/cuda_language_extension.h"
#include "sdfg/conditional_schedule.h"
#include "sdfg/data_flow/library_node.h"

using namespace sdfg;

TEST(BlockDispatcherTest, DispatchNode) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto& block = builder.add_block(root);

    auto final_sdfg = builder.move();

    ConditionalSchedule schedule(final_sdfg);

    codegen::CLanguageExtension language_extension;
    codegen::Instrumentation instrumentation(schedule.schedule(0));
    codegen::BlockDispatcher dispatcher(language_extension, schedule.schedule(0), block,
                                        instrumentation);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::PrettyPrinter library_stream;
    dispatcher.dispatch_node(main_stream, globals_stream, library_stream);

    EXPECT_EQ(library_stream.str(), "");
    EXPECT_EQ(globals_stream.str(), "");
    EXPECT_EQ(main_stream.str(), "");
}

TEST(BlockDispatcherTest, DispatchNode_withDataflow) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    builder.add_container("a", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("b", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("c", types::Scalar(types::PrimitiveType::Int32));

    auto& block = builder.add_block(root);
    auto& access_node_1 = builder.add_access(block, "a");
    auto& access_node_2 = builder.add_access(block, "b");
    auto& access_node_3 = builder.add_access(block, "c");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::add,
                                        {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                        {{"_in1", types::Scalar(types::PrimitiveType::Int32)},
                                         {"_in2", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block, access_node_1, "void", tasklet, "_in1", data_flow::Subset{});
    builder.add_memlet(block, access_node_2, "void", tasklet, "_in2", data_flow::Subset{});
    builder.add_memlet(block, tasklet, "_out", access_node_3, "void", data_flow::Subset{});

    auto final_sdfg = builder.move();

    ConditionalSchedule schedule(final_sdfg);

    codegen::CLanguageExtension language_extension;
    codegen::Instrumentation instrumentation(schedule.schedule(0));
    codegen::BlockDispatcher dispatcher(language_extension, schedule.schedule(0), block,
                                        instrumentation);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::PrettyPrinter library_stream;
    dispatcher.dispatch(main_stream, globals_stream, library_stream);

    EXPECT_EQ(main_stream.str(),
              "{\n    int _in1 = a;\n    int _in2 = b;\n    int _out;\n\n    _out = _in1 + "
              "_in2;\n\n    c = _out;\n}\n");
    EXPECT_EQ(library_stream.str(), "");
    EXPECT_EQ(globals_stream.str(), "");
}

TEST(DataFlowDispatcherTest, DispatchTasklet) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    builder.add_container("a", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("b", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("c", types::Scalar(types::PrimitiveType::Int32));

    auto& block = builder.add_block(root);
    auto& access_node_1 = builder.add_access(block, "a");
    auto& access_node_2 = builder.add_access(block, "b");
    auto& access_node_3 = builder.add_access(block, "c");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::add,
                                        {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                        {{"_in1", types::Scalar(types::PrimitiveType::Int32)},
                                         {"_in2", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block, access_node_1, "void", tasklet, "_in1", data_flow::Subset{});
    builder.add_memlet(block, access_node_2, "void", tasklet, "_in2", data_flow::Subset{});
    builder.add_memlet(block, tasklet, "_out", access_node_3, "void", data_flow::Subset{});

    auto final_sdfg = builder.move();

    ConditionalSchedule schedule(final_sdfg);

    codegen::CLanguageExtension language_extension;
    codegen::DataFlowDispatcher dispatcher(language_extension, schedule.schedule(0).sdfg(),
                                           block.dataflow());

    codegen::PrettyPrinter main_stream;
    dispatcher.dispatch(main_stream);

    EXPECT_EQ(main_stream.str(),
              "{\n    int _in1 = a;\n    int _in2 = b;\n    int _out;\n\n    _out = _in1 + "
              "_in2;\n\n    c = _out;\n}\n");
}

TEST(DataFlowDispatcherTest, DispatchLibraryNodebarrier_local) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto& block = builder.add_block(root);

    auto& library_node =
        builder.add_library_node(block, data_flow::LibraryNodeCode::barrier_local, {}, {});

    auto final_sdfg = builder.move();

    ConditionalSchedule schedule(final_sdfg);

    codegen::CUDALanguageExtension language_extension;
    codegen::DataFlowDispatcher dispatcher(language_extension, schedule.schedule(0).sdfg(),
                                           block.dataflow());

    codegen::PrettyPrinter main_stream;
    dispatcher.dispatch(main_stream);

    EXPECT_EQ(main_stream.str(), "{\n\n    __syncthreads();\n}\n");
}

/*
TEST(DataFlowDispatcherTest, DispatchRef_Input)
{
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    builder.add_container("a", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("b", types::Pointer(types::Scalar(types::PrimitiveType::Int32)));

    auto& block = builder.add_block(root);
    auto& access_node_1 = builder.add_access(block, "a");
    auto& access_node_2 = builder.add_access(block, "b");
    builder.add_memlet(block, access_node_1, "refs", access_node_2, "void", data_flow::Subset{});

    auto final_sdfg = builder.move();

    ConditionalSchedule schedule(final_sdfg);

    codegen::CLanguageExtension language_extension;
    codegen::DataFlowDispatcher dispatcher(
        language_extension,
        schedule.schedule(0).sdfg(),
        block.dataflow()
    );

    codegen::PrettyPrinter main_stream;
    dispatcher.dispatch(main_stream);

    EXPECT_EQ(main_stream.str(), "{\n    b = &a;\n}\n");
}

TEST(DataFlowDispatcherTest, DispatchRef_InputWithSubset)
{
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    builder.add_container("a", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("b",
types::Array(types::Pointer(types::Scalar(types::PrimitiveType::Int32)), symbolic::integer(10)));

    auto& block = builder.add_block(root);
    auto& access_node_1 = builder.add_access(block, "a");
    auto& access_node_2 = builder.add_access(block, "b");
    builder.add_memlet(block, access_node_1, "refs", access_node_2, "void",
data_flow::Subset{symbolic::integer(2)});

    auto final_sdfg = builder.move();

    ConditionalSchedule schedule(final_sdfg);

    codegen::CLanguageExtension language_extension;
    codegen::DataFlowDispatcher dispatcher(
        language_extension,
        schedule.schedule(0).sdfg(),
        block.dataflow()
    );

    codegen::PrettyPrinter main_stream;
    dispatcher.dispatch(main_stream);

    EXPECT_EQ(main_stream.str(), "{\n    b[2] = &a;\n}\n");
}

TEST(DataFlowDispatcherTest, DispatchRef_Output)
{
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    builder.add_container("a", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("b", types::Pointer(types::Scalar(types::PrimitiveType::Int32)));

    auto& block = builder.add_block(root);
    auto& access_node_1 = builder.add_access(block, "a");
    auto& access_node_2 = builder.add_access(block, "b");
    builder.add_memlet(block, access_node_1, "void", access_node_2, "refs", data_flow::Subset{});

    auto final_sdfg = builder.move();

    ConditionalSchedule schedule(final_sdfg);

    codegen::CLanguageExtension language_extension;
    codegen::DataFlowDispatcher dispatcher(
        language_extension,
        schedule.schedule(0).sdfg(),
        block.dataflow()
    );

    codegen::PrettyPrinter main_stream;
    dispatcher.dispatch(main_stream);

    EXPECT_EQ(main_stream.str(), "{\n    b = &a;\n}\n");
}

TEST(DataFlowDispatcherTest, DispatchRef_OutputWithSubset)
{
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    builder.add_container("a", types::Array(types::Scalar(types::PrimitiveType::Int32),
symbolic::integer(10))); builder.add_container("b",
types::Pointer(types::Scalar(types::PrimitiveType::Int32)));

    auto& block = builder.add_block(root);
    auto& access_node_1 = builder.add_access(block, "a");
    auto& access_node_2 = builder.add_access(block, "b");
    builder.add_memlet(block, access_node_1, "void", access_node_2, "refs",
data_flow::Subset{symbolic::integer(1)});

    auto final_sdfg = builder.move();

    ConditionalSchedule schedule(final_sdfg);

    codegen::CLanguageExtension language_extension;
    codegen::DataFlowDispatcher dispatcher(
        language_extension,
        schedule.schedule(0).sdfg(),
        block.dataflow()
    );

    codegen::PrettyPrinter main_stream;
    dispatcher.dispatch(main_stream);

    EXPECT_EQ(main_stream.str(), "{\n    b = &a[1];\n}\n");
}

TEST(DataFlowDispatcherTest, DispatchRef_Nullptr)
{
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    builder.add_container("b", types::Pointer(types::Scalar(types::PrimitiveType::Int32)));

    auto& block = builder.add_block(root);
    auto& access_node_1 = builder.add_access(block, "nullptr");
    auto& access_node_2 = builder.add_access(block, "b");
    builder.add_memlet(block, access_node_1, "refs", access_node_2, "void", data_flow::Subset{});

    auto final_sdfg = builder.move();

    ConditionalSchedule schedule(final_sdfg);

    codegen::CLanguageExtension language_extension;
    codegen::DataFlowDispatcher dispatcher(
        language_extension,
        schedule.schedule(0).sdfg(),
        block.dataflow()
    );

    codegen::PrettyPrinter main_stream;
    dispatcher.dispatch(main_stream);

    EXPECT_EQ(main_stream.str(), "{\n    b = nullptr;\n}\n");
}

TEST(DataFlowDispatcherTest, DispatchRef_Nullptr_Subset)
{
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    builder.add_container("b",
types::Array(types::Pointer(types::Scalar(types::PrimitiveType::Int32)), symbolic::integer(10)));

    auto& block = builder.add_block(root);
    auto& access_node_1 = builder.add_access(block, "nullptr");
    auto& access_node_2 = builder.add_access(block, "b");
    builder.add_memlet(block, access_node_1, "refs", access_node_2, "void",
data_flow::Subset{symbolic::integer(0)});

    auto final_sdfg = builder.move();

    ConditionalSchedule schedule(final_sdfg);

    codegen::CLanguageExtension language_extension;
    codegen::DataFlowDispatcher dispatcher(
        language_extension,
        schedule.schedule(0).sdfg(),
        block.dataflow()
    );

    codegen::PrettyPrinter main_stream;
    dispatcher.dispatch(main_stream);

    EXPECT_EQ(main_stream.str(), "{\n    b[0] = nullptr;\n}\n");
}
*/
