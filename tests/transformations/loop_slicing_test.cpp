#include "sdfg/transformations/loop_slicing.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/passes/pipeline.h"

using namespace sdfg;

TEST(LoopSlicingTest, FirstIteration) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add slicing
    auto& if_else = builder.add_if_else(body);
    auto& branch_1 = builder.add_case(if_else, symbolic::Eq(indvar, symbolic::integer(0)));
    auto& branch_2 = builder.add_case(if_else, symbolic::Ne(indvar, symbolic::integer(0)));

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    // Apply
    transformations::LoopSlicing transformation(loop);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    // Check
    auto new_loop = dynamic_cast<structured_control_flow::For*>(&root.at(0).first);
    EXPECT_NE(new_loop, nullptr);
    EXPECT_TRUE(symbolic::eq(new_loop->init(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(new_loop->condition(), symbolic::Lt(new_loop->indvar(), symbolic::integer(1))));

    auto new_loop2 = dynamic_cast<structured_control_flow::For*>(&root.at(1).first);
    EXPECT_NE(new_loop2, nullptr);
    EXPECT_TRUE(symbolic::eq(new_loop2->init(), symbolic::integer(1)));
    EXPECT_TRUE(symbolic::eq(new_loop2->condition(), symbolic::Lt(new_loop2->indvar(), symbolic::symbol("N"))));
}
