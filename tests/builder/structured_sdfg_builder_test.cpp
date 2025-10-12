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

TEST(SDFG2StructuredSDFGTest, Sequence) {
    builder::SDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& state1 = builder.add_state(true);
    auto& state2 = builder.add_state_after(state1);
    auto& state3 = builder.add_return_state_after(state2, "");

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder structured_builder(*sdfg);
    auto structured_sdfg = structured_builder.move();

    auto& root = structured_sdfg->root();
    EXPECT_EQ(root.size(), 4);

    auto block1 = root.at(0);
    EXPECT_TRUE(dynamic_cast<const structured_control_flow::Block*>(&block1.first));
    EXPECT_EQ(block1.second.size(), 0);

    auto block2 = root.at(1);
    EXPECT_TRUE(dynamic_cast<const structured_control_flow::Block*>(&block2.first));
    EXPECT_EQ(block2.second.size(), 0);

    auto block3 = root.at(2);
    EXPECT_TRUE(dynamic_cast<const structured_control_flow::Block*>(&block3.first));
    EXPECT_EQ(block3.second.size(), 0);

    auto ret = root.at(3);
    EXPECT_TRUE(dynamic_cast<const structured_control_flow::Return*>(&ret.first));
    EXPECT_EQ(ret.second.size(), 0);
}

TEST(SDFG2StructuredSDFGTest, IfElse) {
    builder::SDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int64);
    builder.add_container("i", desc);

    auto& init_state = builder.add_state(true);
    auto& if_state = builder.add_state();
    auto& else_state = builder.add_state();
    auto& end_state = builder.add_return_state("");
    builder.add_edge(
        init_state,
        if_state,
        {{symbolic::symbol("i"), symbolic::integer(0)}},
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        DebugInfo()
    );
    builder.add_edge(
        init_state,
        else_state,
        {{symbolic::symbol("i"), symbolic::integer(1)}},
        symbolic::Ge(symbolic::symbol("i"), symbolic::integer(10)),
        DebugInfo()
    );
    builder.add_edge(if_state, end_state);
    builder.add_edge(else_state, end_state);

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder structured_builder(*sdfg);
    auto structured_sdfg = structured_builder.move();

    // IfElse
    auto& root = structured_sdfg->root();
    EXPECT_EQ(root.size(), 4);

    auto init_block = root.at(0);
    EXPECT_TRUE(dynamic_cast<const structured_control_flow::Block*>(&init_block.first));
    EXPECT_EQ(init_block.second.size(), 0);

    auto if_else = root.at(1);
    EXPECT_TRUE(dynamic_cast<const structured_control_flow::IfElse*>(&if_else.first));
    EXPECT_EQ(if_else.second.size(), 0);

    {
        auto if_case = dynamic_cast<const structured_control_flow::IfElse&>(if_else.first).at(0);
        EXPECT_TRUE(symbolic::eq(if_case.second, symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10))));
        EXPECT_EQ(if_case.first.size(), 2);

        auto if_case_first_block = if_case.first.at(0);
        EXPECT_TRUE(dynamic_cast<const structured_control_flow::Block*>(&if_case_first_block.first));
        EXPECT_EQ(if_case_first_block.second.size(), 1);
        EXPECT_TRUE(symbolic::eq(if_case_first_block.second.assignments().at(symbolic::symbol("i")), symbolic::integer(0))
        );

        auto if_case_second_block = if_case.first.at(1);
        EXPECT_TRUE(dynamic_cast<const structured_control_flow::Block*>(&if_case_second_block.first));
        EXPECT_EQ(if_case_second_block.second.size(), 0);

        auto else_case = dynamic_cast<const structured_control_flow::IfElse&>(if_else.first).at(1);
        EXPECT_TRUE(symbolic::eq(else_case.second, symbolic::Ge(symbolic::symbol("i"), symbolic::integer(10))));
        EXPECT_EQ(else_case.first.size(), 2);

        auto else_case_first_block = else_case.first.at(0);
        EXPECT_TRUE(dynamic_cast<const structured_control_flow::Block*>(&else_case_first_block.first));
        EXPECT_EQ(else_case_first_block.second.size(), 1);
        EXPECT_TRUE(symbolic::
                        eq(else_case_first_block.second.assignments().at(symbolic::symbol("i")), symbolic::integer(1)));

        auto else_case_second_block = else_case.first.at(1);
        EXPECT_TRUE(dynamic_cast<const structured_control_flow::Block*>(&else_case_second_block.first));
        EXPECT_EQ(else_case_second_block.second.size(), 0);
    }

    auto pdom_block = root.at(2);
    EXPECT_TRUE(dynamic_cast<const structured_control_flow::Block*>(&pdom_block.first));
    EXPECT_EQ(pdom_block.second.size(), 0);

    auto ret_block = root.at(3);
    EXPECT_TRUE(dynamic_cast<const structured_control_flow::Return*>(&ret_block.first));
    EXPECT_EQ(ret_block.second.size(), 0);
}

TEST(SDFG2StructuredSDFGTest, While) {
    builder::SDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int64);
    builder.add_container("i", desc);

    auto iter_sym = symbolic::symbol("i");
    auto init_expr = symbolic::integer(0);
    auto cond_expr = symbolic::Lt(iter_sym, symbolic::integer(10));
    auto update_expr = symbolic::add(iter_sym, symbolic::integer(1));

    auto& init_state = builder.add_state(true);
    auto loop = builder.add_loop(init_state, iter_sym, init_expr, cond_expr, update_expr);
    auto& end_state = builder.add_return_state_after(std::get<2>(loop), "");

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder structured_builder(*sdfg);
    auto structured_sdfg = structured_builder.move();

    // Loop
    auto& root = structured_sdfg->root();
    EXPECT_EQ(root.size(), 6);

    auto child1 = root.at(0);
    EXPECT_TRUE(dynamic_cast<const structured_control_flow::Block*>(&child1.first));
    EXPECT_EQ(child1.second.size(), 1);
    EXPECT_TRUE(symbolic::eq(child1.second.assignments().at(iter_sym), init_expr));

    auto child2 = root.at(1);
    EXPECT_TRUE(dynamic_cast<const structured_control_flow::Block*>(&child2.first));
    EXPECT_EQ(child2.second.size(), 0);

    auto child3 = root.at(2);
    EXPECT_TRUE(dynamic_cast<const structured_control_flow::IfElse*>(&child3.first));
    EXPECT_EQ(child3.second.size(), 0);

    {
        auto if_case = dynamic_cast<const structured_control_flow::IfElse&>(child3.first).at(0);
        EXPECT_TRUE(symbolic::eq(if_case.second, cond_expr->logical_not()));
        EXPECT_EQ(if_case.first.size(), 1);

        auto if_case_child1 = if_case.first.at(0);
        EXPECT_TRUE(dynamic_cast<const structured_control_flow::Block*>(&if_case_child1.first));
        EXPECT_EQ(if_case_child1.second.size(), 0);

        auto else_case = dynamic_cast<const structured_control_flow::IfElse&>(child3.first).at(1);
        EXPECT_TRUE(symbolic::eq(else_case.second, cond_expr));
        EXPECT_EQ(else_case.first.size(), 1);

        auto else_case_child2 = else_case.first.at(0);
        EXPECT_TRUE(dynamic_cast<const structured_control_flow::While*>(&else_case_child2.first));
        EXPECT_EQ(else_case_child2.second.size(), 0);
    }

    auto child4 = root.at(3);
    EXPECT_TRUE(dynamic_cast<const structured_control_flow::Block*>(&child4.first));
    EXPECT_EQ(child4.second.size(), 0);

    auto child5 = root.at(4);
    EXPECT_TRUE(dynamic_cast<const structured_control_flow::Block*>(&child5.first));
    EXPECT_EQ(child5.second.size(), 0);

    auto child6 = root.at(5);
    EXPECT_TRUE(dynamic_cast<const structured_control_flow::Return*>(&child6.first));
    EXPECT_EQ(child6.second.size(), 0);
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

TEST(StructuredSDFGBuilderTest, addForEach) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Pointer opaque_desc;
    builder.add_container("list", opaque_desc, true);
    builder.add_container("iter", opaque_desc);

    auto sym_list = symbolic::symbol("list");
    auto sym_iter = symbolic::symbol("iter");

    auto& root = builder.subject().root();
    EXPECT_EQ(root.element_id(), 0);

    /**
     * Doubled linked list with start and end 'list'
     * iter: {
     *  next: ptr;
     *  value: ...
     * }
     * for (auto iter = *list; iter != list; iter = *iter) {
     * 
     * }
     */

    auto& scope = builder.add_for_each(
        root,
        sym_iter,
        sym_list,
        sym_iter,
        sym_list
    );
    EXPECT_EQ(scope.element_id(), 1);
    EXPECT_EQ(scope.root().element_id(), 2);
    EXPECT_EQ(root.at(0).second.element_id(), 3);

    EXPECT_EQ(scope.has_init(), true);
    EXPECT_TRUE(symbolic::eq(scope.init(), sym_list));
    EXPECT_TRUE(symbolic::eq(scope.iterator(), sym_iter));
    EXPECT_TRUE(symbolic::eq(scope.end(), sym_list));
    EXPECT_TRUE(symbolic::eq(scope.update(), sym_iter));

    auto sdfg = builder.move();

    EXPECT_EQ(sdfg->root().size(), 1);
    auto child = sdfg->root().at(0);
    EXPECT_EQ(&child.first, &scope);
    EXPECT_EQ(child.second.size(), 0);
}

TEST(StructuredSDFGBuilderTest, addForEach_Transition) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Pointer opaque_desc;
    builder.add_container("list", opaque_desc, true);
    builder.add_container("iter", opaque_desc);

    types::Scalar int_desc(types::PrimitiveType::Int64);
    builder.add_container("N", int_desc, true);

    auto sym_list = symbolic::symbol("list");
    auto sym_iter = symbolic::symbol("iter");

    auto& root = builder.subject().root();
    EXPECT_EQ(root.element_id(), 0);

    /**
     * Doubled linked list with start and end 'list'
     * iter: {
     *  next: ptr;
     *  value: ...
     * }
     * for (auto iter = *list; iter != list; iter = *iter) {
     * 
     * }
     */

    auto& scope = builder.add_for_each(
        root,
        sym_iter,
        sym_list,
        sym_iter,
        sym_list,
        {{symbolic::symbol("N"), symbolic::zero()}}
    );
    EXPECT_EQ(scope.element_id(), 1);
    EXPECT_EQ(scope.root().element_id(), 2);
    EXPECT_EQ(root.at(0).second.element_id(), 3);

    EXPECT_EQ(scope.has_init(), true);
    EXPECT_TRUE(symbolic::eq(scope.init(), sym_list));
    EXPECT_TRUE(symbolic::eq(scope.iterator(), sym_iter));
    EXPECT_TRUE(symbolic::eq(scope.end(), sym_list));
    EXPECT_TRUE(symbolic::eq(scope.update(), sym_iter));

    auto sdfg = builder.move();

    EXPECT_EQ(sdfg->root().size(), 1);
    auto child = sdfg->root().at(0);
    EXPECT_EQ(&child.first, &scope);
    EXPECT_EQ(child.second.size(), 1);
    EXPECT_TRUE(symbolic::eq(child.second.assignments().at(symbolic::symbol("N")), symbolic::zero()));
}

TEST(StructuredSDFGBuilderTest, addForEachBefore) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Pointer opaque_desc;
    builder.add_container("list", opaque_desc, true);
    builder.add_container("iter", opaque_desc);

    types::Scalar int_desc(types::PrimitiveType::Int64);
    builder.add_container("N", int_desc, true);

    auto sym_list = symbolic::symbol("list");
    auto sym_iter = symbolic::symbol("iter");

    auto& root = builder.subject().root();
    EXPECT_EQ(root.element_id(), 0);

    auto& block_base =
        builder.add_block(root, control_flow::Assignments{{symbolic::symbol("N"), SymEngine::integer(10)}});
    
    auto& scope = builder.add_for_each_before(
        root,
        block_base,
        sym_iter,
        sym_list,
        sym_iter,
        SymEngine::null,
        {{symbolic::symbol("N"), symbolic::zero()}}
    );

    EXPECT_EQ(scope.has_init(), false);
    EXPECT_TRUE(symbolic::eq(scope.iterator(), sym_iter));
    EXPECT_TRUE(symbolic::eq(scope.end(), sym_list));
    EXPECT_TRUE(symbolic::eq(scope.update(), sym_iter));

    auto sdfg = builder.move();

    auto child = sdfg->root().at(0);
    EXPECT_EQ(&child.first, &scope);
    EXPECT_EQ(child.second.size(), 1);
    EXPECT_TRUE(symbolic::eq(child.second.assignments().at(symbolic::symbol("N")), symbolic::zero()));
}

TEST(StructuredSDFGBuilderTest, addForEachAfter) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Pointer opaque_desc;
    builder.add_container("list", opaque_desc, true);
    builder.add_container("iter", opaque_desc);

    types::Scalar int_desc(types::PrimitiveType::Int64);
    builder.add_container("N", int_desc, true);

    auto sym_list = symbolic::symbol("list");
    auto sym_iter = symbolic::symbol("iter");

    auto& root = builder.subject().root();
    EXPECT_EQ(root.element_id(), 0);

    auto& block_base =
        builder.add_block(root, control_flow::Assignments{{symbolic::symbol("N"), SymEngine::integer(10)}});
    auto& block_base2 =
        builder.add_block(root, control_flow::Assignments{{symbolic::symbol("N"), SymEngine::integer(10)}});
    
    /**
     * Doubled linked list with start and end 'list'
     * iter: {
     *  next: ptr;
     *  value: ...
     * }
     * for (auto iter = *list; iter != list; iter = *iter) {
     * 
     * }
     */

    auto& scope = builder.add_for_each_after(
        root,
        block_base,
        sym_iter,
        sym_list,
        sym_iter,
        SymEngine::null,
        {{symbolic::symbol("N"), symbolic::zero()}}
    );

    EXPECT_EQ(scope.has_init(), false);
    EXPECT_TRUE(symbolic::eq(scope.iterator(), sym_iter));
    EXPECT_TRUE(symbolic::eq(scope.end(), sym_list));
    EXPECT_TRUE(symbolic::eq(scope.update(), sym_iter));

    auto sdfg = builder.move();

    auto child = sdfg->root().at(1);
    EXPECT_EQ(&child.first, &scope);
    EXPECT_EQ(child.second.size(), 1);
    EXPECT_TRUE(symbolic::eq(child.second.assignments().at(symbolic::symbol("N")), symbolic::zero()));
}
