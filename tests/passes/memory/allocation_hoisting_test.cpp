#include "sdfg/passes/memory/allocation_hoisting.h"

#include <gtest/gtest.h>

#include "sdfg/data_flow/library_nodes/stdlib/stdlib.h"

using namespace sdfg;

TEST(AllocationHoistingPassTest, Malloc_Argument) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Pointer opaque_desc;
    builder.add_container("arg0", opaque_desc, true);

    auto& block = builder.add_block(root);
    auto& access_node = builder.add_access(block, "arg0");
    auto& lib_node = builder.add_library_node<stdlib::MallocNode>(block, DebugInfo(), symbolic::integer(1024));
    builder.add_computational_memlet(block, lib_node, "_ret", access_node, {}, opaque_desc, DebugInfo());

    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::AllocationHoistingPass pass_;
    EXPECT_TRUE(pass_.run(builder, analysis_manager));

    EXPECT_EQ(block.dataflow().nodes().size(), 0);
    EXPECT_EQ(block.dataflow().edges().size(), 0);

    auto& type = builder.subject().type("arg0");
    EXPECT_TRUE(type.storage_type().is_cpu_heap());
    EXPECT_TRUE(symbolic::eq(type.storage_type().allocation_size(), symbolic::integer(1024)));
    EXPECT_EQ(type.storage_type().allocation_lifetime(), types::StorageType::AllocationLifetime::Lifetime_Default);
}

TEST(AllocationHoistingPassTest, Malloc_Transient) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Pointer opaque_desc;
    builder.add_container("tmp", opaque_desc);

    auto& block = builder.add_block(root);
    auto& access_node = builder.add_access(block, "tmp");
    auto& lib_node = builder.add_library_node<stdlib::MallocNode>(block, DebugInfo(), symbolic::integer(1024));
    builder.add_computational_memlet(block, lib_node, "_ret", access_node, {}, opaque_desc, DebugInfo());

    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::AllocationHoistingPass pass_;
    EXPECT_TRUE(pass_.run(builder, analysis_manager));

    EXPECT_EQ(block.dataflow().nodes().size(), 0);
    EXPECT_EQ(block.dataflow().edges().size(), 0);

    auto& type = builder.subject().type("tmp");
    EXPECT_TRUE(type.storage_type().is_cpu_heap());
    EXPECT_TRUE(symbolic::eq(type.storage_type().allocation_size(), symbolic::integer(1024)));
    EXPECT_EQ(type.storage_type().allocation_lifetime(), types::StorageType::AllocationLifetime::Lifetime_Default);
}

TEST(AllocationHoistingPassTest, Alloca_Transient) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Pointer opaque_desc;
    builder.add_container("tmp", opaque_desc);

    auto& block = builder.add_block(root);
    auto& access_node = builder.add_access(block, "tmp");
    auto& lib_node = builder.add_library_node<stdlib::AllocaNode>(block, DebugInfo(), symbolic::integer(1024));
    builder.add_computational_memlet(block, lib_node, "_ret", access_node, {}, opaque_desc, DebugInfo());

    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::AllocationHoistingPass pass_;
    EXPECT_TRUE(pass_.run(builder, analysis_manager));

    EXPECT_EQ(block.dataflow().nodes().size(), 0);
    EXPECT_EQ(block.dataflow().edges().size(), 0);

    auto& type = builder.subject().type("tmp");
    EXPECT_TRUE(type.storage_type().is_cpu_stack());
    EXPECT_TRUE(symbolic::eq(type.storage_type().allocation_size(), symbolic::integer(1024)));
    EXPECT_EQ(type.storage_type().allocation_lifetime(), types::StorageType::AllocationLifetime::Lifetime_Default);
}
