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

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("B", opaque_desc, true);

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
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a, tasklet, "_in", {indvar}, desc);
    builder.add_computational_memlet(block, tasklet, "_out", b, {indvar}, desc);

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
    EXPECT_TRUE(symbolic::eq(map->indvar(), symbolic::symbol("i")));
}

TEST(For2MapTest, MultiBound) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("B", opaque_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::And(symbolic::Lt(indvar, bound), symbolic::Le(indvar, symbolic::symbol("M")));
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& a = builder.add_access(block, "A");
    auto& b = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a, tasklet, "_in", {indvar}, desc);
    builder.add_computational_memlet(block, tasklet, "_out", b, {indvar}, desc);

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
    EXPECT_TRUE(symbolic::eq(map->indvar(), symbolic::symbol("i")));
}

TEST(For2MapTest, NonContiguousDomain) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("B", opaque_desc, true);

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
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a, tasklet, "_in", {indvar}, desc);
    builder.add_computational_memlet(block, tasklet, "_out", b, {indvar}, desc);

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
    EXPECT_TRUE(symbolic::eq(map->indvar(), symbolic::symbol("i")));
}

TEST(For2MapTest, NonCanonicalBound) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("B", opaque_desc, true);

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
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a, tasklet, "_in", {indvar}, desc);
    builder.add_computational_memlet(block, tasklet, "_out", b, {indvar}, desc);

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
    EXPECT_TRUE(symbolic::eq(map->indvar(), symbolic::symbol("i")));
}

TEST(For2MapTest, Shift) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("B", opaque_desc, true);

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
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a, tasklet, "_in", {indvar}, desc);
    builder.add_computational_memlet(block, tasklet, "_out", b, {indvar}, desc);

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
    EXPECT_TRUE(symbolic::eq(map->indvar(), symbolic::symbol("i")));
}

TEST(For2MapTest, LastValue) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("B", opaque_desc, true);

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
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a, tasklet, "_in", {indvar}, desc);
    builder.add_computational_memlet(block, tasklet, "_out", b, {indvar}, desc);

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
    EXPECT_TRUE(symbolic::eq(map->indvar(), symbolic::symbol("i")));
}

TEST(For2MapTest, Tiled) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("i_tile", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i_tile");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto tile_size = symbolic::integer(8);
    auto update = symbolic::add(indvar, tile_size);

    auto& loop_outer = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop_outer.root();

    auto indvar_tile = symbolic::symbol("i");
    auto init_tile = indvar;
    auto condition_tile = symbolic::
        And(symbolic::Lt(indvar_tile, symbolic::symbol("N")),
            symbolic::Lt(indvar_tile, symbolic::add(indvar, tile_size)));
    auto update_tile = symbolic::add(indvar_tile, symbolic::one());

    auto& loop_inner = builder.add_for(body, indvar_tile, condition_tile, init_tile, update_tile);
    auto& body_inner = loop_inner.root();

    // Add computation
    auto& block = builder.add_block(body_inner);
    auto& a_in = builder.add_access(block, "A");
    auto& one_node = builder.add_constant(block, "1.0", base_desc);
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in1", {symbolic::symbol("i")}, desc);
    builder.add_computational_memlet(block, one_node, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", a_out, {symbolic::symbol("i")}, desc);

    // Analysis
    auto sdfg_opt = builder.move();
    builder::StructuredSDFGBuilder builder_opt(sdfg_opt);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::For2MapPass conversion_pass;
    EXPECT_TRUE(conversion_pass.run(builder_opt, analysis_manager));
    EXPECT_TRUE(conversion_pass.run(builder_opt, analysis_manager));

    auto& sdfg_map = builder_opt.subject();

    // Check
    auto map_outer = dynamic_cast<const structured_control_flow::Map*>(&sdfg_map.root().at(0).first);
    EXPECT_TRUE(map_outer != nullptr);
    EXPECT_TRUE(symbolic::eq(map_outer->indvar(), symbolic::symbol("i_tile")));

    auto map_inner = dynamic_cast<const structured_control_flow::Map*>(&map_outer->root().at(0).first);
    EXPECT_TRUE(map_inner != nullptr);
    EXPECT_TRUE(symbolic::eq(map_inner->indvar(), symbolic::symbol("i")));
}

TEST(For2MapTest, NonContiguousMemory) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Pointer desc2(static_cast<const types::IType&>(desc));

    types::Pointer opaque_desc;
    builder.add_container("B_", opaque_desc, true);
    builder.add_container("B", opaque_desc);
    builder.add_container("A", opaque_desc, true);

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

    // Add Dereference
    {
        auto& block = builder.add_block(body);
        auto& b_ = builder.add_access(block, "B_");
        auto& b = builder.add_access(block, "B");
        builder.add_dereference_memlet(block, b_, b, true, desc2);
    }

    // Add computation
    {
        auto& block = builder.add_block(body);
        auto& a = builder.add_access(block, "A");
        auto& b = builder.add_access(block, "B");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, a, tasklet, "_in", {indvar}, desc);
        builder.add_computational_memlet(block, tasklet, "_out", b, {indvar}, desc);
    }

    // Analysis
    auto sdfg_opt = builder.move();
    builder::StructuredSDFGBuilder builder_opt(sdfg_opt);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::For2MapPass conversion_pass;
    EXPECT_FALSE(conversion_pass.run(builder_opt, analysis_manager));
}
