#include "sdfg/passes/structured_control_flow/for2map.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/symbolic/symbolic.h"

using namespace sdfg;

TEST(For2MapTest, Basic) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("B", desc, true);

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

    // Add computation
    auto& block = builder.add_block(body);
    auto& a = builder.add_access(block, "A");
    auto& b = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", base_desc},
                                        {{"_in", base_desc}});
    builder.add_memlet(block, a, "void", tasklet, "_in", {indvar});
    builder.add_memlet(block, tasklet, "_out", b, "void", {indvar});

    // Analysis
    auto sdfg_opt = builder.move();
    builder::StructuredSDFGBuilder builder_opt(sdfg_opt);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::For2MapPass conversion_pass;
    EXPECT_TRUE(conversion_pass.run(builder_opt, analysis_manager));

    auto& sdfg_map = builder_opt.subject();

    // Check
    auto map = dynamic_cast<const structured_control_flow::Map*>(&sdfg_map.root().at(0).first);
    EXPECT_TRUE(map != nullptr);
    EXPECT_TRUE(symbolic::eq(map->num_iterations(), symbolic::symbol("N")));
    EXPECT_TRUE(symbolic::eq(map->indvar(), symbolic::symbol("i")));
}

TEST(For2MapTest, MultiBound) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("B", desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition =
        symbolic::And(symbolic::Lt(indvar, bound), symbolic::Le(indvar, symbolic::symbol("M")));
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& a = builder.add_access(block, "A");
    auto& b = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", base_desc},
                                        {{"_in", base_desc}});
    builder.add_memlet(block, a, "void", tasklet, "_in", {indvar});
    builder.add_memlet(block, tasklet, "_out", b, "void", {indvar});

    // Analysis
    auto sdfg_opt = builder.move();
    builder::StructuredSDFGBuilder builder_opt(sdfg_opt);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::For2MapPass conversion_pass;
    EXPECT_TRUE(conversion_pass.run(builder_opt, analysis_manager));

    auto& sdfg_map = builder_opt.subject();

    // Check
    auto map = dynamic_cast<const structured_control_flow::Map*>(&sdfg_map.root().at(0).first);
    EXPECT_TRUE(map != nullptr);
    EXPECT_TRUE(symbolic::eq(map->num_iterations(),
                             symbolic::min(symbolic::symbol("N"),
                                           symbolic::add(symbolic::symbol("M"), symbolic::one()))));
    EXPECT_TRUE(symbolic::eq(map->indvar(), symbolic::symbol("i")));
}

TEST(For2MapTest, NonContiguous) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("B", desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(2));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& a = builder.add_access(block, "A");
    auto& b = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", base_desc},
                                        {{"_in", base_desc}});
    builder.add_memlet(block, a, "void", tasklet, "_in", {indvar});
    builder.add_memlet(block, tasklet, "_out", b, "void", {indvar});

    // Analysis
    auto sdfg_opt = builder.move();
    builder::StructuredSDFGBuilder builder_opt(sdfg_opt);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::For2MapPass conversion_pass;
    EXPECT_FALSE(conversion_pass.run(builder_opt, analysis_manager));
}

TEST(For2MapTest, NonCanonicalBound) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("B", desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Ge(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& a = builder.add_access(block, "A");
    auto& b = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", base_desc},
                                        {{"_in", base_desc}});
    builder.add_memlet(block, a, "void", tasklet, "_in", {indvar});
    builder.add_memlet(block, tasklet, "_out", b, "void", {indvar});

    // Analysis
    auto sdfg_opt = builder.move();
    builder::StructuredSDFGBuilder builder_opt(sdfg_opt);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::For2MapPass conversion_pass;
    EXPECT_FALSE(conversion_pass.run(builder_opt, analysis_manager));
}

TEST(For2MapTest, Shift) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("B", desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::one();
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& a = builder.add_access(block, "A");
    auto& b = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", base_desc},
                                        {{"_in", base_desc}});
    auto& memlet_1 = builder.add_memlet(block, a, "void", tasklet, "_in", {indvar});
    auto& memlet_2 = builder.add_memlet(block, tasklet, "_out", b, "void", {indvar});

    // Analysis
    auto sdfg_opt = builder.move();
    builder::StructuredSDFGBuilder builder_opt(sdfg_opt);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::For2MapPass conversion_pass;
    EXPECT_TRUE(conversion_pass.run(builder_opt, analysis_manager));

    auto& sdfg_map = builder_opt.subject();

    // Check
    auto map = dynamic_cast<const structured_control_flow::Map*>(&sdfg_map.root().at(0).first);
    EXPECT_TRUE(map != nullptr);
    EXPECT_TRUE(
        symbolic::eq(map->num_iterations(), symbolic::sub(symbolic::symbol("N"), symbolic::one())));
    EXPECT_TRUE(symbolic::eq(map->indvar(), symbolic::symbol("i")));

    EXPECT_TRUE(symbolic::eq(memlet_1.subset().at(0),
                             symbolic::add(symbolic::symbol("i"), symbolic::one())));
    EXPECT_TRUE(symbolic::eq(memlet_2.subset().at(0),
                             symbolic::add(symbolic::symbol("i"), symbolic::one())));
}

TEST(For2MapTest, LastValue) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("B", desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::one();
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& a = builder.add_access(block, "A");
    auto& b = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", base_desc},
                                        {{"_in", base_desc}});
    auto& memlet_1 = builder.add_memlet(block, a, "void", tasklet, "_in", {indvar});
    auto& memlet_2 = builder.add_memlet(block, tasklet, "_out", b, "void", {indvar});

    // Analysis
    auto sdfg_opt = builder.move();
    builder::StructuredSDFGBuilder builder_opt(sdfg_opt);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::For2MapPass conversion_pass;
    EXPECT_TRUE(conversion_pass.run(builder_opt, analysis_manager));

    auto& sdfg_map = builder_opt.subject();

    // Check
    auto map = dynamic_cast<const structured_control_flow::Map*>(&sdfg_map.root().at(0).first);
    EXPECT_TRUE(map != nullptr);
    EXPECT_TRUE(
        symbolic::eq(map->num_iterations(), symbolic::sub(symbolic::symbol("N"), symbolic::one())));
    EXPECT_TRUE(symbolic::eq(map->indvar(), symbolic::symbol("i")));

    auto& transition = sdfg_map.root().at(1).second;
    EXPECT_TRUE(transition.assignments().size() == 1);
    EXPECT_TRUE(transition.assignments().find(indvar) != transition.assignments().end());
    EXPECT_TRUE(symbolic::eq(transition.assignments().at(indvar), symbolic::symbol("N")));
}
