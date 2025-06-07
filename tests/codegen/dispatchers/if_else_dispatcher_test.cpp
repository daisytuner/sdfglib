#include "sdfg/codegen/dispatchers/if_else_dispatcher.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/codegen/language_extensions/c_language_extension.h"

using namespace sdfg;

TEST(IfElseDispatcherTest, DispatchNode_Trivial) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto& if_else = builder.add_if_else(root);
    auto& case_1 = builder.add_case(if_else, symbolic::__true__());
    auto& case_2 = builder.add_case(if_else, symbolic::__false__());

    auto final_sdfg = builder.move();

    codegen::CLanguageExtension language_extension;
    codegen::Instrumentation instrumentation(*final_sdfg);
    codegen::IfElseDispatcher dispatcher(language_extension, *final_sdfg, if_else, instrumentation);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::PrettyPrinter library_stream;
    dispatcher.dispatch_node(main_stream, globals_stream, library_stream);

    EXPECT_EQ(globals_stream.str(), "");
    EXPECT_EQ(library_stream.str(), "");
    EXPECT_EQ(main_stream.str(), "if(true)\n{\n}\nelse if(false)\n{\n}\n");
}

TEST(IfElseDispatcherTest, DispatchNode) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    builder.add_container("a", types::Scalar(types::PrimitiveType::Int8));

    auto& if_else = builder.add_if_else(root);
    auto& case_1 =
        builder.add_case(if_else, symbolic::Eq(symbolic::symbol("a"), symbolic::integer(0)));
    auto& case_2 =
        builder.add_case(if_else, symbolic::Ne(symbolic::symbol("a"), symbolic::integer(0)));

    auto final_sdfg = builder.move();

    codegen::CLanguageExtension language_extension;
    codegen::Instrumentation instrumentation(*final_sdfg);
    codegen::IfElseDispatcher dispatcher(language_extension, *final_sdfg, if_else, instrumentation);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::PrettyPrinter library_stream;
    dispatcher.dispatch_node(main_stream, globals_stream, library_stream);

    EXPECT_EQ(globals_stream.str(), "");
    EXPECT_EQ(library_stream.str(), "");
    EXPECT_EQ(main_stream.str(), "if((0 == a))\n{\n}\nelse if((0 != a))\n{\n}\n");
}
