#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/codegen/language_extensions/c_language_extension.h"
#include "sdfg/conditional_schedule.h"

using namespace sdfg;

TEST(NodeDispatcherTest, BeginNode_Declaration) {
    builder::StructuredSDFGBuilder builder("sdfg_a");
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    builder.add_container("b", types::Pointer(types::Scalar(types::PrimitiveType::Int32)));

    auto& block = builder.add_block(root);

    auto final_sdfg = builder.move();

    ConditionalSchedule schedule(final_sdfg);
    schedule.schedule(0).allocation_lifetime("b", &block);
    schedule.schedule(0).allocation_type("b", AllocationType::DECLARE);

    codegen::CLanguageExtension language_extension;
    codegen::BlockDispatcher dispatcher(language_extension, schedule.schedule(0), block, false);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::PrettyPrinter library_stream;
    dispatcher.dispatch(main_stream, globals_stream, library_stream);

    EXPECT_EQ(globals_stream.str(), "");
    EXPECT_EQ(library_stream.str(), "");
    EXPECT_EQ(main_stream.str(), "{\n    int (*b);\n}\n");
}

TEST(NodeDispatcherTest, BeginNode_Allocation) {
    builder::StructuredSDFGBuilder builder("sdfg_a");
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    builder.add_container("b", types::Pointer(types::Scalar(types::PrimitiveType::Int32)));

    auto& block = builder.add_block(root);

    auto final_sdfg = builder.move();

    ConditionalSchedule schedule(final_sdfg);
    schedule.schedule(0).allocation_lifetime("b", &block);
    schedule.schedule(0).allocation_type("b", AllocationType::ALLOCATE);

    codegen::CLanguageExtension language_extension;
    codegen::BlockDispatcher dispatcher(language_extension, schedule.schedule(0), block, false);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::PrettyPrinter library_stream;
    dispatcher.dispatch(main_stream, globals_stream, library_stream);

    EXPECT_EQ(globals_stream.str(), "");
    EXPECT_EQ(library_stream.str(), "");
    EXPECT_EQ(main_stream.str(),
              "{\n    int (*b) = (int (*)) malloc(1 * sizeof(int ));\n    free(b);\n}\n");
}
