#include <gtest/gtest.h>

#include <ostream>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/passes/pipeline.h"

using namespace sdfg;

TEST(DeadDataEliminationTest, Unused) {
    builder::StructuredSDFGBuilder builder("sdfg");

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);
    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::DeadDataElimination pass;
    bool applied = true;
    do {
        applied = pass.run(builder_opt, analysis_manager);
    } while (applied);
    sdfg = builder_opt.move();

    // Check result
    EXPECT_EQ(sdfg->root().size(), 0);
    EXPECT_EQ(sdfg->containers().size(), 0);
}

TEST(DeadDataEliminationTest, WriteWithoutRead_Transition) {
    builder::StructuredSDFGBuilder builder("sdfg");

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);
    auto sym1 = symbolic::symbol("i");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root, {{sym1, symbolic::integer(0)}});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::DeadDataElimination pass;
    bool applied = true;
    do {
        applied = pass.run(builder_opt, analysis_manager);
    } while (applied);
    sdfg = builder_opt.move();

    // Check result
    EXPECT_EQ(sdfg->root().size(), 1);
    auto child1 = sdfg->root().at(0);
    EXPECT_EQ(child1.second.assignments().size(), 0);

    EXPECT_EQ(sdfg->containers().size(), 0);
}

TEST(DeadDataEliminationTest, WriteWithoutRead_Dataflow) {
    builder::StructuredSDFGBuilder builder("sdfg");

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("j", desc);

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);

    auto& output_node = builder.add_access(block, "j");
    auto& tasklet =
        builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", desc}, {{"0", desc}});
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {symbolic::integer(0)});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::DeadDataElimination pass;
    bool applied = true;
    do {
        applied = pass.run(builder_opt, analysis_manager);
    } while (applied);
    sdfg = builder_opt.move();

    // Check result
    EXPECT_EQ(sdfg->root().size(), 1);
    auto& block1 = static_cast<const structured_control_flow::Block&>(sdfg->root().at(0).first);
    EXPECT_EQ(block1.dataflow().nodes().size(), 0);
    EXPECT_EQ(block1.dataflow().edges().size(), 0);

    EXPECT_EQ(sdfg->containers().size(), 0);
}

TEST(DeadDataEliminationTest, WriteAfterWrite_For) {
    builder::StructuredSDFGBuilder builder("sdfg");

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);
    auto sym = symbolic::symbol("i");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root, {{sym, symbolic::integer(0)}});
    auto& loop = builder.add_for(root, sym, symbolic::Lt(sym, symbolic::integer(10)),
                                 symbolic::integer(0), symbolic::add(sym, symbolic::integer(1)));

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::DeadDataElimination pass;
    bool applied = true;
    do {
        applied = pass.run(builder_opt, analysis_manager);
    } while (applied);
    sdfg = builder_opt.move();

    // Check result
    EXPECT_EQ(sdfg->root().size(), 2);
    auto child0 = sdfg->root().at(0);
    EXPECT_EQ(child0.second.assignments().size(), 0);

    EXPECT_EQ(sdfg->containers().size(), 1);
}

TEST(DeadDataEliminationTest, WriteAfterWrite_WhileBody) {
    builder::StructuredSDFGBuilder builder("sdfg");

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);
    builder.add_container("N", desc, true);
    auto sym = symbolic::symbol("i");
    auto symN = symbolic::symbol("N");

    auto& root = builder.subject().root();
    auto& before = builder.add_block(root, {{sym, symbolic::integer(0)}});

    auto& loop = builder.add_while(root);
    auto& block1 = builder.add_block(loop.root(), {{sym, symbolic::integer(0)}});
    auto& block2 = builder.add_block(loop.root(), {{symN, sym}});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::DeadDataElimination pass;
    bool applied = true;
    do {
        applied = pass.run(builder_opt, analysis_manager);
    } while (applied);
    sdfg = builder_opt.move();

    // Check result
    auto child0 = sdfg->root().at(0);
    EXPECT_EQ(child0.second.assignments().size(), 0);

    auto& child1 = static_cast<structured_control_flow::While&>(sdfg->root().at(1).first);
    auto& body = child1.root();
    EXPECT_EQ(body.at(0).second.assignments().size(), 1);
    EXPECT_EQ(body.at(1).second.assignments().size(), 1);

    EXPECT_EQ(sdfg->containers().size(), 2);
}

TEST(DeadDataEliminationTest, WriteAfterWrite_ClosedBranches) {
    builder::StructuredSDFGBuilder builder("sdfg");

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);
    builder.add_container("N", desc, true);
    auto sym = symbolic::symbol("i");
    auto symN = symbolic::symbol("N");

    auto& root = builder.subject().root();

    auto& before = builder.add_block(root, {{sym, symbolic::integer(0)}});

    auto& if_else = builder.add_if_else(root);
    auto& case1 = builder.add_case(if_else, symbolic::__true__());
    auto& case2 = builder.add_case(if_else, symbolic::__false__());
    auto& block1 = builder.add_block(case1, {{sym, symbolic::integer(0)}});
    auto& block2 = builder.add_block(case2, {{sym, symbolic::integer(1)}});

    auto& after = builder.add_block(root, {{symN, sym}});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::DeadDataElimination pass;
    bool applied = true;
    do {
        applied = pass.run(builder_opt, analysis_manager);
    } while (applied);
    sdfg = builder_opt.move();

    // Check result
    auto child0 = sdfg->root().at(0);
    EXPECT_EQ(child0.second.assignments().size(), 0);

    auto child2 = sdfg->root().at(2);
    EXPECT_EQ(child2.second.assignments().size(), 1);

    EXPECT_EQ(sdfg->containers().size(), 2);
}

TEST(DeadDataEliminationTest, WriteAfterWrite_OpenBranches) {
    builder::StructuredSDFGBuilder builder("sdfg");

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);
    builder.add_container("N", desc, true);
    auto sym = symbolic::symbol("i");
    auto symN = symbolic::symbol("N");

    auto& root = builder.subject().root();

    auto& before = builder.add_block(root, {{sym, symbolic::integer(0)}});

    auto& if_else = builder.add_if_else(root);
    auto& case1 = builder.add_case(if_else, symbolic::__true__());
    auto& case2 = builder.add_case(if_else, symbolic::__false__());
    auto& block1 = builder.add_block(case1, {{sym, symbolic::integer(0)}});
    auto& block2 = builder.add_block(case2, {});

    auto& after = builder.add_block(root, {{symN, sym}});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::DeadDataElimination pass;
    bool applied = true;
    do {
        applied = pass.run(builder_opt, analysis_manager);
    } while (applied);
    sdfg = builder_opt.move();

    // Check result
    auto child0 = sdfg->root().at(0);
    EXPECT_EQ(child0.second.assignments().size(), 1);

    auto child2 = sdfg->root().at(2);
    EXPECT_EQ(child2.second.assignments().size(), 1);

    EXPECT_EQ(sdfg->containers().size(), 2);
}

TEST(DeadDataEliminationTest, WriteAfterWrite_IncompleteBranches) {
    builder::StructuredSDFGBuilder builder("sdfg");

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);
    builder.add_container("N", desc, true);
    auto sym = symbolic::symbol("i");
    auto symN = symbolic::symbol("N");

    auto& root = builder.subject().root();

    auto& before = builder.add_block(root, {{sym, symbolic::integer(0)}});

    auto& if_else = builder.add_if_else(root);
    auto& case1 = builder.add_case(if_else, symbolic::Lt(symN, symbolic::integer(10)));
    auto& block1 = builder.add_block(case1, {{sym, symbolic::integer(0)}});

    auto& after = builder.add_block(root, {{symN, sym}});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::DeadDataElimination pass;
    bool applied = true;
    do {
        applied = pass.run(builder_opt, analysis_manager);
    } while (applied);
    sdfg = builder_opt.move();

    // Check result
    auto child0 = sdfg->root().at(0);
    EXPECT_EQ(child0.second.assignments().size(), 1);

    auto child2 = sdfg->root().at(2);
    EXPECT_EQ(child2.second.assignments().size(), 1);

    EXPECT_EQ(sdfg->containers().size(), 2);
}

TEST(DeadDataEliminationTest, WriteAfterWrite_ContinueBreak_NoReads) {
    builder::StructuredSDFGBuilder builder("sdfg");

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);
    builder.add_container("N", desc, true);
    auto sym = symbolic::symbol("i");
    auto symN = symbolic::symbol("N");

    auto& root = builder.subject().root();
    auto& loop = builder.add_while(root);
    auto& body = loop.root();

    auto& before = builder.add_block(body, {{sym, symbolic::integer(0)}});

    auto& if_else = builder.add_if_else(body);
    auto& case1 = builder.add_case(if_else, symbolic::Lt(sym, symbolic::integer(10)));
    auto& block1 = builder.add_block(case1, {{sym, symbolic::integer(10)}});
    auto& cont1 = builder.add_continue(case1, loop);
    auto& case2 = builder.add_case(if_else, symbolic::Ge(sym, symbolic::integer(10)));
    auto& block2 = builder.add_block(case2, {{sym, symbolic::integer(0)}});
    auto& break1 = builder.add_break(case2, loop);

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::DeadDataElimination pass;
    bool applied = true;
    do {
        applied = pass.run(builder_opt, analysis_manager);
    } while (applied);
    sdfg = builder_opt.move();

    // Check result
    auto& new_loop = static_cast<structured_control_flow::While&>(sdfg->root().at(0).first);
    auto& new_body = new_loop.root();

    auto child1 = new_body.at(0);
    EXPECT_EQ(child1.second.assignments().size(), 1);

    auto& child2 = static_cast<structured_control_flow::IfElse&>(new_body.at(1).first);
    auto case1_1 = child2.at(0).first.at(0);
    EXPECT_EQ(case1_1.second.assignments().size(), 0);

    auto case2_1 = child2.at(1).first.at(0);
    EXPECT_EQ(case2_1.second.assignments().size(), 0);

    EXPECT_EQ(sdfg->containers().size(), 2);
}

TEST(DeadDataEliminationTest, WriteAfterWrite_ContinueBreak_Read) {
    builder::StructuredSDFGBuilder builder("sdfg");

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);
    builder.add_container("N", desc, true);
    auto sym = symbolic::symbol("i");
    auto symN = symbolic::symbol("N");

    auto& root = builder.subject().root();
    auto& loop = builder.add_while(root);
    auto& body = loop.root();

    auto& before = builder.add_block(body, {{sym, symbolic::integer(0)}});

    auto& if_else = builder.add_if_else(body);
    auto& case1 = builder.add_case(if_else, symbolic::Lt(sym, symbolic::integer(10)));
    auto& block1 = builder.add_block(case1, {{sym, symbolic::integer(10)}});
    auto& cont1 = builder.add_continue(case1, loop);
    auto& case2 = builder.add_case(if_else, symbolic::Ge(sym, symbolic::integer(10)));
    auto& block2 = builder.add_block(case2, {{sym, symbolic::integer(0)}});
    auto& break1 = builder.add_break(case2, loop);

    auto& after2 = builder.add_block(root, {{symN, sym}});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::DeadDataElimination pass;
    bool applied = true;
    do {
        applied = pass.run(builder_opt, analysis_manager);
    } while (applied);
    sdfg = builder_opt.move();

    // Check result
    auto& new_loop = static_cast<structured_control_flow::While&>(sdfg->root().at(0).first);
    auto& new_body = new_loop.root();

    auto child1 = new_body.at(0);
    EXPECT_EQ(child1.second.assignments().size(), 1);

    auto& child2 = static_cast<structured_control_flow::IfElse&>(new_body.at(1).first);

    // Over-approximation. Can be zero when analysis becomes more precise.
    auto case1_1 = child2.at(0).first.at(0);
    EXPECT_EQ(case1_1.second.assignments().size(), 1);

    auto case2_1 = child2.at(1).first.at(0);
    EXPECT_EQ(case2_1.second.assignments().size(), 1);

    auto child3 = sdfg->root().at(1);
    EXPECT_EQ(child3.second.assignments().size(), 1);

    EXPECT_EQ(sdfg->containers().size(), 2);
}

TEST(DeadDataEliminationTest, WriteAfterWrite_ContinueBreak_OpenRead) {
    builder::StructuredSDFGBuilder builder("sdfg");

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);
    builder.add_container("N", desc, true);
    auto sym = symbolic::symbol("i");
    auto symN = symbolic::symbol("N");

    auto& root = builder.subject().root();
    auto& loop = builder.add_while(root);
    auto& body = loop.root();

    auto& if_else = builder.add_if_else(body);
    auto& case1 = builder.add_case(if_else, symbolic::Lt(sym, symbolic::integer(10)));
    auto& block1 = builder.add_block(case1, {{sym, symbolic::integer(10)}});
    auto& cont1 = builder.add_continue(case1, loop);
    auto& case2 = builder.add_case(if_else, symbolic::Ge(sym, symbolic::integer(10)));
    auto& block2 = builder.add_block(case2, {{sym, symbolic::integer(0)}});
    auto& break1 = builder.add_break(case2, loop);

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::DeadDataElimination pass;
    bool applied = true;
    do {
        applied = pass.run(builder_opt, analysis_manager);
    } while (applied);
    sdfg = builder_opt.move();

    // Check result
    auto& new_loop = static_cast<structured_control_flow::While&>(sdfg->root().at(0).first);
    auto& new_body = new_loop.root();

    auto& child1 = static_cast<structured_control_flow::IfElse&>(new_body.at(0).first);

    auto case1_1 = child1.at(0).first.at(0);
    EXPECT_EQ(case1_1.second.assignments().size(), 1);

    // Over-approximation. Can be zero when analysis becomes more precise.
    auto case2_1 = child1.at(1).first.at(0);
    EXPECT_EQ(case2_1.second.assignments().size(), 1);

    EXPECT_EQ(sdfg->containers().size(), 2);
}
