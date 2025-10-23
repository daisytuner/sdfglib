#include "sdfg/passes/memory/allocation_management.h"

#include <gtest/gtest.h>

#include "sdfg/data_flow/library_nodes/stdlib/stdlib.h"

using namespace sdfg;

TEST(AllocationManagementPassTest, Malloc_Argument) {
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
    passes::AllocationManagementPass pass_;
    EXPECT_TRUE(pass_.run(builder, analysis_manager));

    EXPECT_EQ(block.dataflow().nodes().size(), 0);
    EXPECT_EQ(block.dataflow().edges().size(), 0);

    auto& type = builder.subject().type("arg0");
    EXPECT_TRUE(type.storage_type().is_cpu_heap());
    EXPECT_TRUE(symbolic::eq(type.storage_type().allocation_size(), symbolic::integer(1024)));
    EXPECT_EQ(type.storage_type().allocation(), types::StorageType::AllocationType::Managed);
    EXPECT_EQ(type.storage_type().deallocation(), types::StorageType::AllocationType::Unmanaged);
}

TEST(AllocationManagementPassTest, Malloc_Transient) {
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
    passes::AllocationManagementPass pass_;
    EXPECT_TRUE(pass_.run(builder, analysis_manager));

    EXPECT_EQ(block.dataflow().nodes().size(), 0);
    EXPECT_EQ(block.dataflow().edges().size(), 0);

    auto& type = builder.subject().type("tmp");
    EXPECT_TRUE(type.storage_type().is_cpu_heap());
    EXPECT_TRUE(symbolic::eq(type.storage_type().allocation_size(), symbolic::integer(1024)));
    EXPECT_EQ(type.storage_type().allocation(), types::StorageType::AllocationType::Managed);
    EXPECT_EQ(type.storage_type().deallocation(), types::StorageType::AllocationType::Unmanaged);
}

TEST(AllocationManagementPassTest, Alloca_Transient) {
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
    passes::AllocationManagementPass pass_;
    EXPECT_TRUE(pass_.run(builder, analysis_manager));

    EXPECT_EQ(block.dataflow().nodes().size(), 0);
    EXPECT_EQ(block.dataflow().edges().size(), 0);

    auto& type = builder.subject().type("tmp");
    EXPECT_TRUE(type.storage_type().is_cpu_stack());
    EXPECT_TRUE(symbolic::eq(type.storage_type().allocation_size(), symbolic::integer(1024)));
    EXPECT_EQ(type.storage_type().allocation(), types::StorageType::AllocationType::Managed);
    EXPECT_EQ(type.storage_type().deallocation(), types::StorageType::AllocationType::Unmanaged);
}

TEST(AllocationManagementPassTest, Free_Argument) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Pointer opaque_desc;
    builder.add_container("arg0", opaque_desc, true);

    auto& block = builder.add_block(root);
    auto& access_node_in = builder.add_access(block, "arg0");
    auto& access_node_out = builder.add_access(block, "arg0");
    auto& lib_node = builder.add_library_node<stdlib::FreeNode>(block, DebugInfo());
    builder.add_computational_memlet(block, access_node_in, lib_node, "_ptr", {}, opaque_desc, DebugInfo());
    builder.add_computational_memlet(block, lib_node, "_ptr", access_node_out, {}, opaque_desc, DebugInfo());

    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::AllocationManagementPass pass_;
    EXPECT_TRUE(pass_.run(builder, analysis_manager));

    EXPECT_EQ(block.dataflow().nodes().size(), 0);
    EXPECT_EQ(block.dataflow().edges().size(), 0);

    auto& type = builder.subject().type("arg0");
    EXPECT_TRUE(type.storage_type().is_cpu_heap());
    EXPECT_TRUE(type.storage_type().allocation_size().is_null());
    EXPECT_EQ(type.storage_type().allocation(), types::StorageType::AllocationType::Unmanaged);
    EXPECT_EQ(type.storage_type().deallocation(), types::StorageType::AllocationType::Managed);
}

TEST(AllocationManagementPassTest, Free_Transient) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Pointer opaque_desc;
    builder.add_container("tmp", opaque_desc);

    auto& block = builder.add_block(root);
    auto& access_node_in = builder.add_access(block, "tmp");
    auto& access_node_out = builder.add_access(block, "tmp");
    auto& lib_node = builder.add_library_node<stdlib::FreeNode>(block, DebugInfo());
    builder.add_computational_memlet(block, access_node_in, lib_node, "_ptr", {}, opaque_desc, DebugInfo());
    builder.add_computational_memlet(block, lib_node, "_ptr", access_node_out, {}, opaque_desc, DebugInfo());

    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::AllocationManagementPass pass_;
    EXPECT_TRUE(pass_.run(builder, analysis_manager));

    EXPECT_EQ(block.dataflow().nodes().size(), 0);
    EXPECT_EQ(block.dataflow().edges().size(), 0);

    auto& type = builder.subject().type("tmp");
    EXPECT_TRUE(type.storage_type().is_cpu_heap());
    EXPECT_TRUE(type.storage_type().allocation_size().is_null());
    EXPECT_EQ(type.storage_type().allocation(), types::StorageType::AllocationType::Unmanaged);
    EXPECT_EQ(type.storage_type().deallocation(), types::StorageType::AllocationType::Managed);
}
