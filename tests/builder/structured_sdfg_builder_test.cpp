#include "sdfg/builder/structured_sdfg_builder.h"

#include <gtest/gtest.h>

#include "sdfg/builder/sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/barrier_local_node.h"
#include "sdfg/element.h"
#include "sdfg/symbolic/symbolic.h"

using namespace sdfg;

TEST(StructuredSDFGBuilderTest, Empty) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto sdfg = builder.move();

    EXPECT_EQ(sdfg->name(), "sdfg_1");
    EXPECT_EQ(sdfg->root().size(), 0);
}

TEST(StructuredSDFGBuilderTest, AddBlock) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int64);
    builder.add_container("N", desc);

    auto& root = builder.subject().root();
    EXPECT_EQ(root.element_id(), 0);

    auto& block = builder.add_block(root, control_flow::Assignments{{symbolic::symbol("N"), SymEngine::integer(10)}});
    EXPECT_EQ(block.element_id(), 1);
    EXPECT_EQ(root.at(0).second.element_id(), 2);

    auto sdfg = builder.move();

    EXPECT_EQ(sdfg->name(), "sdfg_1");
    EXPECT_EQ(sdfg->root().size(), 1);

    auto child = sdfg->root().at(0);
    EXPECT_EQ(&child.first, &block);
    EXPECT_EQ(child.second.size(), 1);
}

TEST(StructuredSDFGBuilderTest, AddBlockBefore) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int64);
    builder.add_container("N", desc);

    auto& root = builder.subject().root();
    EXPECT_EQ(root.element_id(), 0);

    auto& block_base =
        builder.add_block(root, control_flow::Assignments{{symbolic::symbol("N"), SymEngine::integer(10)}});
    EXPECT_EQ(block_base.element_id(), 1);
    EXPECT_EQ(root.at(0).second.element_id(), 2);

    auto& block = builder.add_block_before(root, block_base, {}, {});
    EXPECT_EQ(block.element_id(), 3);
    EXPECT_EQ(root.at(0).second.element_id(), 4);

    auto sdfg = builder.move();

    EXPECT_EQ(sdfg->name(), "sdfg_1");
    EXPECT_EQ(sdfg->root().size(), 2);

    auto child = sdfg->root().at(0);
    EXPECT_EQ(&child.first, &block);
    EXPECT_EQ(child.second.size(), 0);
}

TEST(StructuredSDFGBuilderTest, AddBlockAfter) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int64);
    builder.add_container("N", desc);

    auto& root = builder.subject().root();
    EXPECT_EQ(root.element_id(), 0);

    auto& block_base =
        builder.add_block(root, control_flow::Assignments{{symbolic::symbol("N"), SymEngine::integer(10)}});
    EXPECT_EQ(block_base.element_id(), 1);
    EXPECT_EQ(root.at(0).second.element_id(), 2);

    auto& block_base2 =
        builder.add_block(root, control_flow::Assignments{{symbolic::symbol("N"), SymEngine::integer(10)}});
    EXPECT_EQ(block_base2.element_id(), 3);
    EXPECT_EQ(root.at(1).second.element_id(), 4);

    auto& block = builder.add_block_after(root, block_base, {}, {});
    EXPECT_EQ(block.element_id(), 5);
    EXPECT_EQ(root.at(1).second.element_id(), 6);

    auto sdfg = builder.move();

    EXPECT_EQ(sdfg->name(), "sdfg_1");
    EXPECT_EQ(sdfg->root().size(), 3);

    auto child = sdfg->root().at(1);
    EXPECT_EQ(&child.first, &block);
    EXPECT_EQ(child.second.size(), 0);
}

TEST(StructuredSDFGBuilderTest, AddLibraryNode) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int64);
    builder.add_container("N", desc);

    auto& root = builder.subject().root();
    EXPECT_EQ(root.element_id(), 0);

    auto& block = builder.add_block(root, control_flow::Assignments{{symbolic::symbol("N"), SymEngine::integer(10)}});
    EXPECT_EQ(block.element_id(), 1);
    EXPECT_EQ(root.at(0).second.element_id(), 2);

    auto& lib_node = builder.add_library_node<data_flow::BarrierLocalNode>(block, DebugInfo());
    EXPECT_EQ(lib_node.element_id(), 3);

    auto sdfg = builder.move();

    EXPECT_EQ(sdfg->name(), "sdfg_1");
    EXPECT_EQ(sdfg->root().size(), 1);

    auto child = sdfg->root().at(0);
    EXPECT_EQ(&child.first, &block);
    EXPECT_EQ(child.second.size(), 1);

    EXPECT_EQ(block.dataflow().nodes().size(), 1);
    EXPECT_EQ(block.dataflow().edges().size(), 0);
    EXPECT_EQ(&(*block.dataflow().nodes().begin()), &lib_node);
    EXPECT_EQ(lib_node.code(), data_flow::LibraryNodeType_BarrierLocal);
    EXPECT_TRUE(dynamic_cast<data_flow::BarrierLocalNode*>(&lib_node));
}

TEST(StructuredSDFGBuilderTest, AddIfElse) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& root = builder.subject().root();
    EXPECT_EQ(root.element_id(), 0);

    auto& if_else = builder.add_if_else(root);
    EXPECT_EQ(if_else.element_id(), 1);
    EXPECT_EQ(root.at(0).second.element_id(), 2);

    auto& true_case = builder.add_case(if_else, symbolic::__true__());
    EXPECT_EQ(true_case.element_id(), 3);

    auto& false_case = builder.add_case(if_else, symbolic::__false__());
    EXPECT_EQ(false_case.element_id(), 4);

    auto sdfg = builder.move();

    EXPECT_EQ(sdfg->name(), "sdfg_1");
    EXPECT_EQ(sdfg->root().size(), 1);

    auto child = sdfg->root().at(0);
    EXPECT_EQ(&child.first, &if_else);

    EXPECT_EQ(if_else.size(), 2);
    EXPECT_EQ(&if_else.at(0).first, &true_case);
    EXPECT_TRUE(symbolic::is_true(if_else.at(0).second));
    EXPECT_EQ(&if_else.at(1).first, &false_case);
    EXPECT_TRUE(symbolic::is_false(if_else.at(1).second));
}

TEST(StructuredSDFGBuilderTest, AddIfElseBefore) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    builder.add_container("N", types::Scalar(types::PrimitiveType::Int64), true);

    auto& root = builder.subject().root();
    auto& block_base =
        builder.add_block(root, control_flow::Assignments{{symbolic::symbol("N"), SymEngine::integer(10)}});
    auto& if_else = builder.add_if_else_before(root, block_base, {}, {});
    auto& true_case = builder.add_case(if_else, symbolic::__true__());
    auto& false_case = builder.add_case(if_else, symbolic::__false__());

    auto sdfg = builder.move();

    EXPECT_EQ(sdfg->name(), "sdfg_1");
    EXPECT_EQ(sdfg->root().size(), 2);

    auto child = sdfg->root().at(0);
    EXPECT_EQ(&child.first, &if_else);

    EXPECT_EQ(if_else.size(), 2);
    EXPECT_EQ(&if_else.at(0).first, &true_case);
    EXPECT_TRUE(symbolic::is_true(if_else.at(0).second));
    EXPECT_EQ(&if_else.at(1).first, &false_case);
    EXPECT_TRUE(symbolic::is_false(if_else.at(1).second));
}

TEST(StructuredSDFGBuilderTest, addWhile) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& root = builder.subject().root();
    EXPECT_EQ(root.element_id(), 0);

    auto& scope = builder.add_while(root);
    EXPECT_EQ(scope.element_id(), 1);
    EXPECT_EQ(scope.root().element_id(), 2);
    EXPECT_EQ(root.at(0).second.element_id(), 3);

    auto& body = builder.add_block(scope.root());
    EXPECT_EQ(body.element_id(), 4);

    auto& break_state = builder.add_break(scope.root());
    EXPECT_EQ(break_state.element_id(), 6);

    auto sdfg = builder.move();

    EXPECT_EQ(sdfg->name(), "sdfg_1");
    EXPECT_EQ(sdfg->root().size(), 1);

    auto child = sdfg->root().at(0);
    EXPECT_EQ(&child.first, &scope);
    EXPECT_EQ(scope.root().size(), 2);
}

TEST(StructuredSDFGBuilderTest, addFor) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    builder.add_container("N", types::Scalar(types::PrimitiveType::Int64), true);

    auto& root = builder.subject().root();
    EXPECT_EQ(root.element_id(), 0);

    auto& scope = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );
    EXPECT_EQ(scope.element_id(), 1);
    EXPECT_EQ(scope.root().element_id(), 2);
    EXPECT_EQ(root.at(0).second.element_id(), 3);

    auto& body = builder.add_block(scope.root());
    EXPECT_EQ(body.element_id(), 4);

    auto sdfg = builder.move();

    EXPECT_EQ(sdfg->name(), "sdfg_1");
    EXPECT_EQ(sdfg->root().size(), 1);

    auto child = sdfg->root().at(0);
    EXPECT_EQ(&child.first, &scope);
    EXPECT_EQ(scope.root().size(), 1);
}

TEST(StructuredSDFGBuilderTest, addFor_Transition) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    builder.add_container("i", types::Scalar(types::PrimitiveType::Int64), true);

    auto& root = builder.subject().root();
    EXPECT_EQ(root.element_id(), 0);

    auto& scope = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        {{symbolic::symbol("i"), symbolic::zero()}}
    );
    EXPECT_EQ(scope.element_id(), 1);
    EXPECT_EQ(scope.root().element_id(), 2);
    EXPECT_EQ(root.at(0).second.element_id(), 3);

    auto& body = builder.add_block(scope.root());
    EXPECT_EQ(body.element_id(), 4);

    auto sdfg = builder.move();

    EXPECT_EQ(sdfg->name(), "sdfg_1");
    EXPECT_EQ(sdfg->root().size(), 1);
    EXPECT_EQ(sdfg->root().at(0).second.assignments().size(), 1);

    auto child = sdfg->root().at(0);
    EXPECT_EQ(&child.first, &scope);
    EXPECT_EQ(scope.root().size(), 1);
}

TEST(StructuredSDFGBuilderTest, addMap) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& root = builder.subject().root();
    EXPECT_EQ(root.element_id(), 0);

    auto& scope = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    EXPECT_EQ(scope.element_id(), 1);
    EXPECT_EQ(scope.root().element_id(), 2);
    EXPECT_EQ(root.at(0).second.element_id(), 3);

    auto& body = builder.add_block(scope.root());
    EXPECT_EQ(body.element_id(), 4);

    auto sdfg = builder.move();

    EXPECT_EQ(sdfg->name(), "sdfg_1");
    EXPECT_EQ(sdfg->root().size(), 1);

    auto map = dynamic_cast<structured_control_flow::Map*>(&sdfg->root().at(0).first);
    EXPECT_TRUE(map);
    EXPECT_EQ(sdfg->root().at(0).second.assignments().size(), 0);

    auto child = sdfg->root().at(0);
    EXPECT_EQ(&child.first, &scope);
    EXPECT_EQ(scope.root().size(), 1);
}

TEST(StructuredSDFGBuilderTest, addMap_Transition) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    builder.add_container("i", types::Scalar(types::PrimitiveType::Int64), true);

    auto& root = builder.subject().root();
    EXPECT_EQ(root.element_id(), 0);

    auto& scope = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create(),
        {{symbolic::symbol("i"), symbolic::zero()}}
    );
    EXPECT_EQ(scope.element_id(), 1);
    EXPECT_EQ(scope.root().element_id(), 2);
    EXPECT_EQ(root.at(0).second.element_id(), 3);

    auto& body = builder.add_block(scope.root());
    EXPECT_EQ(body.element_id(), 4);

    auto sdfg = builder.move();

    EXPECT_EQ(sdfg->name(), "sdfg_1");
    EXPECT_EQ(sdfg->root().size(), 1);

    auto map = dynamic_cast<structured_control_flow::Map*>(&sdfg->root().at(0).first);
    EXPECT_TRUE(map);
    EXPECT_EQ(sdfg->root().at(0).second.assignments().size(), 1);

    auto child = sdfg->root().at(0);
    EXPECT_EQ(&child.first, &scope);
    EXPECT_EQ(scope.root().size(), 1);
}

TEST(StructuredSDFGBuilderTest, addForBefore) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    builder.add_container("i", types::Scalar(types::PrimitiveType::Int64));
    builder.add_container("N", types::Scalar(types::PrimitiveType::Int64), true);

    auto& root = builder.subject().root();
    auto& block_base =
        builder.add_block(root, control_flow::Assignments{{symbolic::symbol("N"), SymEngine::integer(10)}});
    auto& scope = builder.add_for_before(
        root,
        block_base,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );
    auto& body = builder.add_block(scope.root());

    auto sdfg = builder.move();

    EXPECT_EQ(sdfg->name(), "sdfg_1");
    EXPECT_EQ(sdfg->root().size(), 2);

    auto child = sdfg->root().at(0);
    EXPECT_EQ(&child.first, &scope);
    EXPECT_EQ(scope.root().size(), 1);
}

TEST(StructuredSDFGBuilderTest, addForAfter) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    builder.add_container("i", types::Scalar(types::PrimitiveType::Int64));
    builder.add_container("N", types::Scalar(types::PrimitiveType::Int64), true);

    auto& root = builder.subject().root();
    auto& block_base =
        builder.add_block(root, control_flow::Assignments{{symbolic::symbol("N"), SymEngine::integer(10)}});
    auto& block_base2 =
        builder.add_block(root, control_flow::Assignments{{symbolic::symbol("N"), SymEngine::integer(10)}});
    auto& scope = builder.add_for_after(
        root,
        block_base,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );
    auto& body = builder.add_block(scope.root());

    auto sdfg = builder.move();

    EXPECT_EQ(sdfg->name(), "sdfg_1");
    EXPECT_EQ(sdfg->root().size(), 3);

    auto child = sdfg->root().at(1);
    EXPECT_EQ(&child.first, &scope);
    EXPECT_EQ(scope.root().size(), 1);
}

TEST(SDFG2StructuredSDFGTest, Function_Definition) {
    builder::SDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& start_state = builder.add_state(true);
    auto& end_state = builder.add_return_state_after(start_state, "");

    types::Scalar desc1(types::PrimitiveType::Double);
    types::Scalar desc2(types::PrimitiveType::Int8);
    builder.add_container("argument_1", desc2, true);
    builder.add_container("argument_2", desc1, true);
    builder.add_container("external_1", desc2, false, true);
    builder.add_container("transient_1", desc1);
    builder.add_container("transient_2", desc2);

    auto sdfg = builder.move();
    EXPECT_EQ(sdfg->assumptions().size(), 3);

    sdfg->add_metadata("key", "value");
    sdfg->add_metadata("key2", "value2");

    builder::StructuredSDFGBuilder structured_builder(*sdfg);
    auto structured_sdfg = structured_builder.move();

    EXPECT_EQ(structured_sdfg->name(), sdfg->name());

    size_t args = sdfg->arguments().size();
    EXPECT_EQ(structured_sdfg->arguments().size(), args);
    for (size_t i = 0; i < args; i++) {
        EXPECT_EQ(structured_sdfg->arguments()[i], sdfg->arguments()[i]);
    }

    size_t exts = sdfg->externals().size();
    EXPECT_EQ(structured_sdfg->externals().size(), exts);
    for (size_t i = 0; i < exts; i++) {
        EXPECT_EQ(structured_sdfg->externals()[i], sdfg->externals()[i]);
    }

    EXPECT_EQ(structured_sdfg->containers().size(), sdfg->containers().size());
    for (auto name : structured_sdfg->containers()) {
        EXPECT_EQ(structured_sdfg->type(name), sdfg->type(name));
    }

    EXPECT_EQ(structured_sdfg->assumptions().size(), 3);

    EXPECT_EQ(structured_sdfg->metadata("key"), "value");
    EXPECT_EQ(structured_sdfg->metadata("key2"), "value2");
}

TEST(StructuredSDFGBuilderTest, FindElementById_Root) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& root = builder.subject().root();

    // Test

    EXPECT_EQ(builder.find_element_by_id(root.element_id()), &root);
}

TEST(StructuredSDFGBuilderTest, FindElementById_Block) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);

    // Test

    EXPECT_EQ(builder.find_element_by_id(block.element_id()), &block);
}
