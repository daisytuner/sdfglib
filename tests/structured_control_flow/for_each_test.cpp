#include <gtest/gtest.h>
#include "sdfg/structured_control_flow/map.h"

#include "sdfg/codegen/dispatchers/for_each_dispatcher.h"
#include "sdfg/codegen/language_extensions/c_language_extension.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

using namespace sdfg;

TEST(ForEachTest, SerializeDeserialize) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Pointer opaque_desc;
    builder.add_container("list", opaque_desc, true);
    builder.add_container("iter", opaque_desc);

    auto sym_iter = symbolic::symbol("iter");
    auto sym_list = symbolic::symbol("list");
    auto sym_nullptr = symbolic::__nullptr__();

    auto& for_each = builder.add_for_each(root, sym_iter, sym_nullptr, sym_iter, sym_list);

    serializer::JSONSerializer serializer;
    auto json = serializer.serialize(sdfg);
    auto deserialized_sdfg = serializer.deserialize(json);

    auto& deserialized_root = deserialized_sdfg->root();
    auto deserialized_for_each = dynamic_cast<const structured_control_flow::ForEach*>(&deserialized_root.at(0).first);
    EXPECT_TRUE(deserialized_for_each != nullptr);
    EXPECT_TRUE(deserialized_for_each->has_init());
    EXPECT_TRUE(symbolic::eq(deserialized_for_each->init(), sym_list));
    EXPECT_TRUE(symbolic::eq(deserialized_for_each->iterator(), sym_iter));
    EXPECT_TRUE(symbolic::eq(deserialized_for_each->end(), sym_nullptr));
    EXPECT_TRUE(symbolic::eq(deserialized_for_each->update(), sym_iter));
}

TEST(ForEachTest, Dispatch) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Pointer opaque_desc;
    builder.add_container("list", opaque_desc, true);
    builder.add_container("iter", opaque_desc);

    auto sym_iter = symbolic::symbol("iter");
    auto sym_list = symbolic::symbol("list");
    auto sym_nullptr = symbolic::__nullptr__();

    auto& for_each = builder.add_for_each(root, sym_iter, sym_nullptr, sym_iter, sym_list);

    codegen::CLanguageExtension language_extension;
    auto instrumentation = codegen::InstrumentationPlan::none(sdfg);
    codegen::ForEachDispatcher dispatcher(language_extension, sdfg, for_each, *instrumentation);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::CodeSnippetFactory library_factory;
    dispatcher.dispatch_node(main_stream, globals_stream, library_factory);

    EXPECT_EQ(globals_stream.str(), "");
    EXPECT_TRUE(library_factory.snippets().empty());
    EXPECT_EQ(main_stream.str(), "for(iter = *((void* *) list);iter != NULL;iter = *((void* *) iter))\n{\n}\n");
}

class ForEachVisitor : public visitor::StructuredSDFGVisitor {
public:
    ForEachVisitor(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        : visitor::StructuredSDFGVisitor(builder, analysis_manager) {}

    bool accept(structured_control_flow::ForEach& node) override { return true; };
};

TEST(StructuredSDFGVisitorTest, ForEach) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Pointer opaque_desc;
    builder.add_container("list", opaque_desc, true);
    builder.add_container("iter", opaque_desc);

    auto sym_iter = symbolic::symbol("iter");
    auto sym_list = symbolic::symbol("list");
    auto sym_nullptr = symbolic::__nullptr__();

    auto& for_each = builder.add_for_each(root, sym_iter, sym_nullptr, sym_iter, sym_list);

    analysis::AnalysisManager analysis_manager(builder.subject());
    ForEachVisitor visitor(builder, analysis_manager);
    EXPECT_TRUE(visitor.visit());
}
