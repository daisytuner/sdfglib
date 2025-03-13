#include "sdfg/passes/normalization/conditional_loop_fissioning.h"

#include <gtest/gtest.h>

using namespace sdfg;

TEST(ConditionalLoopFissioningTest, Basic) {
    builder::StructuredSDFGBuilder builder("sdfg");

    types::Scalar desc(types::PrimitiveType::Double);
    types::Pointer ptr_desc(*desc.clone());
    builder.add_container("A", ptr_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt32);
    builder.add_container("a", sym_desc, true);
    builder.add_container("i", sym_desc);
    auto sym = symbolic::symbol("i");
    auto sym2 = symbolic::symbol("a");

    auto& root = builder.subject().root();

    auto& loop = builder.add_for(root, sym, symbolic::Lt(sym, symbolic::integer(10)),
                                 symbolic::zero(), symbolic::add(sym, symbolic::integer(1)));
    auto& body = loop.root();

    auto& ifelse = builder.add_if_else(body);
    auto& branch = builder.add_case(ifelse, symbolic::Eq(sym2, symbolic::zero()));

    auto& block = builder.add_block(branch);
    auto& output_node = builder.add_access(block, "A");
    auto& tasklet =
        builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", desc}, {{"0", desc}});
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {sym});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::ConditionalLoopFissioningPass pass;
    EXPECT_TRUE(pass.run(builder_opt, analysis_manager));

    sdfg = builder_opt.move();

    auto& new_root = sdfg->root().at(0).first;
    EXPECT_TRUE(dynamic_cast<const structured_control_flow::IfElse*>(&new_root) != nullptr);

    auto& new_ifelse = static_cast<const structured_control_flow::IfElse&>(new_root);
    EXPECT_EQ(new_ifelse.size(), 1);

    auto new_branch = new_ifelse.at(0);
    EXPECT_TRUE(symbolic::eq(new_branch.second, symbolic::Eq(sym2, symbolic::zero())));
    EXPECT_EQ(new_branch.first.size(), 1);
    EXPECT_TRUE(dynamic_cast<const structured_control_flow::For*>(&new_branch.first.at(0).first) !=
                nullptr);

    auto& new_loop = static_cast<const structured_control_flow::For&>(new_branch.first.at(0).first);
    EXPECT_EQ(new_loop.root().size(), 1);
    EXPECT_TRUE(dynamic_cast<const structured_control_flow::Sequence*>(
                    &new_loop.root().at(0).first) != nullptr);

    auto& new_sequence =
        static_cast<const structured_control_flow::Sequence&>(new_loop.root().at(0).first);
    EXPECT_EQ(new_sequence.size(), 1);
    EXPECT_TRUE(dynamic_cast<const structured_control_flow::Block*>(&new_sequence.at(0).first) !=
                nullptr);
}

TEST(ConditionalLoopFissioningTest, Iterator) {
    builder::StructuredSDFGBuilder builder("sdfg");

    types::Scalar desc(types::PrimitiveType::Double);
    types::Pointer ptr_desc(*desc.clone());
    builder.add_container("A", ptr_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt32);
    builder.add_container("a", sym_desc, true);
    builder.add_container("i", sym_desc);
    auto sym = symbolic::symbol("i");
    auto sym2 = symbolic::symbol("a");

    auto& root = builder.subject().root();

    auto& loop = builder.add_for(root, sym, symbolic::Lt(sym, symbolic::integer(10)),
                                 symbolic::zero(), symbolic::add(sym, symbolic::integer(1)));
    auto& body = loop.root();

    auto& ifelse = builder.add_if_else(body);
    auto& branch = builder.add_case(ifelse, symbolic::Eq(sym, symbolic::zero()));

    auto& block = builder.add_block(branch);
    auto& output_node = builder.add_access(block, "A");
    auto& tasklet =
        builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", desc}, {{"0", desc}});
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {sym});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::ConditionalLoopFissioningPass pass;
    EXPECT_FALSE(pass.run(builder_opt, analysis_manager));
}
