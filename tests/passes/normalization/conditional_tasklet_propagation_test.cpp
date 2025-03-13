#include "sdfg/passes/normalization/conditional_tasklet_propagation.h"

#include <gtest/gtest.h>

using namespace sdfg;

TEST(ConditionalTaskletPropagationTest, Basic) {
    builder::StructuredSDFGBuilder builder("sdfg");

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);
    builder.add_container("j", desc, true);
    auto sym = symbolic::symbol("i");

    auto& root = builder.subject().root();

    auto& if_else = builder.add_if_else(root);
    auto& branch = builder.add_case(if_else, symbolic::Gt(sym, symbolic::integer(0)));

    auto& block = builder.add_block(branch);
    auto& output_node = builder.add_access(block, "j");
    auto& tasklet =
        builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", desc}, {{"0", desc}});
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {symbolic::integer(0)});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::ConditionalTaskletPropagationPass s2spass;
    EXPECT_TRUE(s2spass.run(builder_opt, analysis_manager));

    sdfg = builder_opt.move();

    auto& first_root = sdfg->root().at(0).first;
    EXPECT_TRUE(dynamic_cast<const structured_control_flow::Sequence*>(&first_root) != nullptr);
    EXPECT_TRUE(symbolic::eq(tasklet.condition(), symbolic::Gt(sym, symbolic::integer(0))));
}

TEST(ConditionalTaskletPropagationTest, WriteConflict) {
    builder::StructuredSDFGBuilder builder("sdfg");

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);
    builder.add_container("j", desc, true);
    auto sym = symbolic::symbol("i");

    auto& root = builder.subject().root();

    auto& if_else = builder.add_if_else(root);
    auto& branch = builder.add_case(if_else, symbolic::Gt(sym, symbolic::integer(0)));

    auto& block = builder.add_block(branch);
    auto& output_node = builder.add_access(block, "j");
    auto& tasklet =
        builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", desc}, {{"0", desc}});
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {symbolic::integer(0)});
    ;

    builder.add_block(branch, {{sym, symbolic::integer(0)}});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::ConditionalTaskletPropagationPass s2spass;
    EXPECT_FALSE(s2spass.run(builder_opt, analysis_manager));
}

TEST(ConditionalTaskletPropagationTest, ReadConflict) {
    builder::StructuredSDFGBuilder builder("sdfg");

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);

    types::Array desc2(desc, symbolic::integer(10));
    builder.add_container("j", desc2, true);
    auto sym = symbolic::symbol("i");

    auto& root = builder.subject().root();

    auto& if_else = builder.add_if_else(root);
    auto& branch = builder.add_case(if_else, symbolic::Gt(sym, symbolic::integer(0)));

    auto& block = builder.add_block(branch);
    auto& output_node = builder.add_access(block, "j");
    auto& tasklet =
        builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", desc}, {{"0", desc}});
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {sym});
    ;

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::ConditionalTaskletPropagationPass s2spass;
    EXPECT_FALSE(s2spass.run(builder_opt, analysis_manager));
}
