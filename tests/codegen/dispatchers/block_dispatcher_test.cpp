#include "sdfg/codegen/dispatchers/block_dispatcher.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/codegen/language_extensions/c_language_extension.h"
#include "sdfg/codegen/language_extensions/cuda_language_extension.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/library_nodes/barrier_local_node.h"

using namespace sdfg;

TEST(BlockDispatcherTest, DispatchNode_Empty) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto& block = builder.add_block(root);

    auto final_sdfg = builder.move();

    codegen::CLanguageExtension language_extension;
    auto instrumentation = codegen::InstrumentationPlan::none(*final_sdfg);
    codegen::BlockDispatcher dispatcher(language_extension, *final_sdfg, block, *instrumentation);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::CodeSnippetFactory library_factory;
    dispatcher.dispatch_node(main_stream, globals_stream, library_factory);

    EXPECT_TRUE(library_factory.snippets().empty());
    EXPECT_EQ(globals_stream.str(), "");
    EXPECT_EQ(main_stream.str(), "");
}

TEST(BlockDispatcherTest, DispatchNode_TopologicalOrder) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    builder.add_container("a", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("b", types::Scalar(types::PrimitiveType::Int32));

    auto& block = builder.add_block(root);
    auto& access_node_in1 = builder.add_access(block, "a");
    auto& access_node_out1 = builder.add_access(block, "b");
    auto& tasklet = builder.add_tasklet(
        block,
        data_flow::TaskletCode::assign,
        {"_out", types::Scalar(types::PrimitiveType::Int32)},
        {{"_in", types::Scalar(types::PrimitiveType::Int32)}}
    );
    builder.add_memlet(block, access_node_in1, "void", tasklet, "_in", data_flow::Subset{});
    builder.add_memlet(block, tasklet, "_out", access_node_out1, "void", data_flow::Subset{});

    auto& access_node_in2 = builder.add_access(block, "b");
    auto& access_node_out2 = builder.add_access(block, "a");
    auto& tasklet_2 = builder.add_tasklet(
        block,
        data_flow::TaskletCode::assign,
        {"_out", types::Scalar(types::PrimitiveType::Int32)},
        {{"_in", types::Scalar(types::PrimitiveType::Int32)}}
    );
    builder.add_memlet(block, access_node_in2, "void", tasklet_2, "_in", data_flow::Subset{});
    builder.add_memlet(block, tasklet_2, "_out", access_node_out2, "void", data_flow::Subset{});

    auto final_sdfg = builder.move();

    codegen::CLanguageExtension language_extension;
    auto instrumentation = codegen::InstrumentationPlan::none(*final_sdfg);
    codegen::BlockDispatcher dispatcher(language_extension, *final_sdfg, block, *instrumentation);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::CodeSnippetFactory library_factory;
    dispatcher.dispatch_node(main_stream, globals_stream, library_factory);

    EXPECT_EQ(
        main_stream.str(),
        "{\n    int _in = b;\n    int _out;\n\n    _out = _in;\n\n    a = _out;\n}\n{\n    int _in = a;\n    int "
        "_out;\n\n    _out = _in;\n\n    b = _out;\n}\n"
    );
    EXPECT_TRUE(library_factory.snippets().empty());
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
    auto& tasklet = builder.add_tasklet(
        block,
        data_flow::TaskletCode::add,
        {"_out", types::Scalar(types::PrimitiveType::Int32)},
        {{"_in1", types::Scalar(types::PrimitiveType::Int32)}, {"_in2", types::Scalar(types::PrimitiveType::Int32)}}
    );
    builder.add_memlet(block, access_node_1, "void", tasklet, "_in1", data_flow::Subset{});
    builder.add_memlet(block, access_node_2, "void", tasklet, "_in2", data_flow::Subset{});
    builder.add_memlet(block, tasklet, "_out", access_node_3, "void", data_flow::Subset{});

    auto final_sdfg = builder.move();

    codegen::CLanguageExtension language_extension;
    codegen::DataFlowDispatcher dispatcher(language_extension, *final_sdfg, block.dataflow());

    codegen::PrettyPrinter main_stream;
    dispatcher.dispatch(main_stream);

    EXPECT_EQ(
        main_stream.str(),
        "{\n    int _in1 = a;\n    int _in2 = b;\n    int _out;\n\n    _out = _in1 + "
        "_in2;\n\n    c = _out;\n}\n"
    );
}

TEST(DataFlowDispatcherTest, DispatchLibraryNode) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto& block = builder.add_block(root);
    builder.add_library_node<sdfg::data_flow::BarrierLocalNode>(block, DebugInfo());

    auto final_sdfg = builder.move();

    codegen::CUDALanguageExtension language_extension;
    codegen::DataFlowDispatcher dispatcher(language_extension, *final_sdfg, block.dataflow());

    codegen::PrettyPrinter main_stream;
    dispatcher.dispatch(main_stream);

    EXPECT_EQ(main_stream.str(), "__syncthreads();\n");
}

TEST(DataFlowDispatcherTest, DispatchRef_Empty) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_type(types::PrimitiveType::Int32);
    types::Pointer pointer_type(base_type);
    types::Pointer pointer_type_2(static_cast<const types::IType&>(pointer_type));

    builder.add_container("a", pointer_type);
    builder.add_container("b", pointer_type_2);

    auto& block = builder.add_block(root);
    auto& access_node_1 = builder.add_access(block, "a");
    auto& access_node_2 = builder.add_access(block, "b");
    builder.add_memlet(block, access_node_1, "void", access_node_2, "ref", data_flow::Subset{});

    auto final_sdfg = builder.move();

    codegen::CLanguageExtension language_extension;
    codegen::DataFlowDispatcher dispatcher(language_extension, *final_sdfg, block.dataflow());

    codegen::PrettyPrinter main_stream;
    dispatcher.dispatch(main_stream);

    EXPECT_EQ(main_stream.str(), "{\n    b = &a;\n}\n");
}

TEST(DataFlowDispatcherTest, DispatchRef_Subset) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_type(types::PrimitiveType::Int32);
    types::Pointer pointer_type(base_type);

    builder.add_container("a", pointer_type);
    builder.add_container("b", pointer_type);

    auto& block = builder.add_block(root);
    auto& access_node_1 = builder.add_access(block, "a");
    auto& access_node_2 = builder.add_access(block, "b");
    builder.add_memlet(block, access_node_1, "void", access_node_2, "ref", data_flow::Subset{symbolic::integer(1)});

    auto final_sdfg = builder.move();

    codegen::CLanguageExtension language_extension;
    codegen::DataFlowDispatcher dispatcher(language_extension, *final_sdfg, block.dataflow());

    codegen::PrettyPrinter main_stream;
    dispatcher.dispatch(main_stream);

    EXPECT_EQ(main_stream.str(), "{\n    b = &a[1];\n}\n");
}

TEST(DataFlowDispatcherTest, DispatchRef_ReinterpretCast) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_type(types::PrimitiveType::Int32);
    types::Pointer pointer_type(base_type);

    types::Scalar base_type_2(types::PrimitiveType::Int64);
    types::Pointer pointer_type_2(base_type_2);

    builder.add_container("a", pointer_type);
    builder.add_container("b", pointer_type_2);

    auto& block = builder.add_block(root);
    auto& access_node_1 = builder.add_access(block, "a");
    auto& access_node_2 = builder.add_access(block, "b");
    builder.add_memlet(block, access_node_1, "void", access_node_2, "ref", data_flow::Subset{symbolic::integer(0)});

    auto final_sdfg = builder.move();

    codegen::CLanguageExtension language_extension;
    codegen::DataFlowDispatcher dispatcher(language_extension, *final_sdfg, block.dataflow());

    codegen::PrettyPrinter main_stream;
    dispatcher.dispatch(main_stream);

    EXPECT_EQ(main_stream.str(), "{\n    b = (long long *) &a[0];\n}\n");
}

TEST(DataFlowDispatcherTest, DispatchRef_Nullptr) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_type(types::PrimitiveType::Int64);
    types::Pointer pointer_type(base_type);

    builder.add_container("b", pointer_type);

    auto& block = builder.add_block(root);
    auto& access_node_1 = builder.add_access(block, symbolic::__nullptr__()->get_name());
    auto& access_node_2 = builder.add_access(block, "b");
    builder.add_memlet(block, access_node_1, "void", access_node_2, "ref", data_flow::Subset{});

    auto final_sdfg = builder.move();

    codegen::CLanguageExtension language_extension;
    codegen::DataFlowDispatcher dispatcher(language_extension, *final_sdfg, block.dataflow());

    codegen::PrettyPrinter main_stream;
    dispatcher.dispatch(main_stream);

    EXPECT_EQ(main_stream.str(), "{\n    b = NULL;\n}\n");
}

TEST(DataFlowDispatcherTest, DispatchRef_Address) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_type(types::PrimitiveType::Int64);
    types::Pointer pointer_type(base_type);

    builder.add_container("b", pointer_type);

    auto& block = builder.add_block(root);
    auto& access_node_1 = builder.add_access(block, "100");
    auto& access_node_2 = builder.add_access(block, "b");
    builder.add_memlet(block, access_node_1, "void", access_node_2, "ref", data_flow::Subset{});

    auto final_sdfg = builder.move();

    codegen::CLanguageExtension language_extension;
    codegen::DataFlowDispatcher dispatcher(language_extension, *final_sdfg, block.dataflow());

    codegen::PrettyPrinter main_stream;
    dispatcher.dispatch(main_stream);

    EXPECT_EQ(main_stream.str(), "{\n    b = (long long *) 100;\n}\n");
}

TEST(DataFlowDispatcherTest, DispatchDeref_Load) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_type(types::PrimitiveType::Int32);
    types::Pointer pointer_type(base_type);
    types::Pointer pointer_type_2(static_cast<const types::IType&>(pointer_type));

    builder.add_container("a", pointer_type_2);
    builder.add_container("b", pointer_type);

    auto& block = builder.add_block(root);
    auto& access_node_1 = builder.add_access(block, "a");
    auto& access_node_2 = builder.add_access(block, "b");
    builder.add_memlet(block, access_node_1, "void", access_node_2, "deref", data_flow::Subset{symbolic::integer(0)});

    auto final_sdfg = builder.move();

    codegen::CLanguageExtension language_extension;
    codegen::DataFlowDispatcher dispatcher(language_extension, *final_sdfg, block.dataflow());

    codegen::PrettyPrinter main_stream;
    dispatcher.dispatch(main_stream);

    EXPECT_EQ(main_stream.str(), "{\n    b = a[0];\n}\n");
}

TEST(DataFlowDispatcherTest, DispatchDeref_Store) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_type(types::PrimitiveType::Int32);
    types::Pointer pointer_type(base_type);
    types::Pointer pointer_type_2(static_cast<const types::IType&>(pointer_type));

    builder.add_container("a", pointer_type);
    builder.add_container("b", pointer_type_2);

    auto& block = builder.add_block(root);
    auto& access_node_1 = builder.add_access(block, "a");
    auto& access_node_2 = builder.add_access(block, "b");
    builder.add_memlet(block, access_node_1, "deref", access_node_2, "void", data_flow::Subset{symbolic::integer(0)});

    auto final_sdfg = builder.move();

    codegen::CLanguageExtension language_extension;
    codegen::DataFlowDispatcher dispatcher(language_extension, *final_sdfg, block.dataflow());

    codegen::PrettyPrinter main_stream;
    dispatcher.dispatch(main_stream);

    EXPECT_EQ(main_stream.str(), "{\n    b[0] = a;\n}\n");
}

TEST(DataFlowDispatcherTest, DispatchDeref_Store_Nullptr) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_type(types::PrimitiveType::Int32);
    types::Pointer pointer_type(base_type);
    types::Pointer pointer_type_2(static_cast<const types::IType&>(pointer_type));

    builder.add_container("a", pointer_type);
    builder.add_container("b", pointer_type_2);

    auto& block = builder.add_block(root);
    auto& access_node_1 = builder.add_access(block, symbolic::__nullptr__()->get_name());
    auto& access_node_2 = builder.add_access(block, "b");
    builder.add_memlet(block, access_node_1, "deref", access_node_2, "void", data_flow::Subset{symbolic::integer(0)});

    auto final_sdfg = builder.move();

    codegen::CLanguageExtension language_extension;
    codegen::DataFlowDispatcher dispatcher(language_extension, *final_sdfg, block.dataflow());

    codegen::PrettyPrinter main_stream;
    dispatcher.dispatch(main_stream);

    EXPECT_EQ(main_stream.str(), "{\n    b[0] = NULL;\n}\n");
}
