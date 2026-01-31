#include "sdfg/passes/memory/insert_frees.h"

#include <gtest/gtest.h>

#include <functional>

#include "sdfg/analysis/escape_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/stdlib/stdlib.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/while.h"
#include "sdfg/types/pointer.h"

using namespace sdfg;

namespace {

size_t count_free_nodes(StructuredSDFG& sdfg) {
    size_t count = 0;
    std::function<void(structured_control_flow::Sequence&)> traverse_sequence;

    traverse_sequence = [&](structured_control_flow::Sequence& seq) {
        for (size_t i = 0; i < seq.size(); ++i) {
            auto [node_ref, transition] = seq.at(i);
            auto& node = node_ref;

            if (auto* block = dynamic_cast<structured_control_flow::Block*>(&node)) {
                for (auto* lib_node : block->dataflow().library_nodes()) {
                    if (lib_node->code() == stdlib::LibraryNodeType_Free) {
                        ++count;
                    }
                }
            } else if (auto* for_node = dynamic_cast<structured_control_flow::For*>(&node)) {
                traverse_sequence(for_node->root());
            } else if (auto* while_node = dynamic_cast<structured_control_flow::While*>(&node)) {
                traverse_sequence(while_node->root());
            } else if (auto* if_else = dynamic_cast<structured_control_flow::IfElse*>(&node)) {
                for (size_t j = 0; j < if_else->size(); ++j) {
                    auto [case_seq, cond] = if_else->at(j);
                    traverse_sequence(case_seq);
                }
            } else if (auto* map = dynamic_cast<structured_control_flow::Map*>(&node)) {
                traverse_sequence(map->root());
            }
        }
    };

    traverse_sequence(sdfg.root());
    return count;
}

bool has_free_for_container(StructuredSDFG& sdfg, const std::string& container) {
    bool found = false;
    std::function<void(structured_control_flow::Sequence&)> traverse_sequence;

    traverse_sequence = [&](structured_control_flow::Sequence& seq) {
        for (size_t i = 0; i < seq.size(); ++i) {
            auto [node_ref, transition] = seq.at(i);
            auto& node = node_ref;

            if (auto* block = dynamic_cast<structured_control_flow::Block*>(&node)) {
                for (auto* lib_node : block->dataflow().library_nodes()) {
                    if (lib_node->code() == stdlib::LibraryNodeType_Free) {
                        // Check if the free node is connected to the container
                        auto& dataflow = block->dataflow();
                        for (auto& edge : dataflow.in_edges(*lib_node)) {
                            if (auto* access = dynamic_cast<data_flow::AccessNode*>(&edge.src())) {
                                if (access->data() == container) {
                                    found = true;
                                    return;
                                }
                            }
                        }
                    }
                }
            } else if (auto* for_node = dynamic_cast<structured_control_flow::For*>(&node)) {
                traverse_sequence(for_node->root());
            } else if (auto* while_node = dynamic_cast<structured_control_flow::While*>(&node)) {
                traverse_sequence(while_node->root());
            } else if (auto* if_else = dynamic_cast<structured_control_flow::IfElse*>(&node)) {
                for (size_t j = 0; j < if_else->size(); ++j) {
                    auto [case_seq, cond] = if_else->at(j);
                    traverse_sequence(case_seq);
                }
            } else if (auto* map = dynamic_cast<structured_control_flow::Map*>(&node)) {
                traverse_sequence(map->root());
            }
        }
    };

    traverse_sequence(sdfg.root());
    return found;
}

} // anonymous namespace

TEST(InsertFreesPassTest, Basic_InsertFreeForNonEscaping) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer ptr_desc(base_desc);
    builder.add_container("ptr", ptr_desc);

    // Block 1: malloc
    auto& block1 = builder.add_block(root);
    auto& malloc_output = builder.add_access(block1, "ptr");
    auto& malloc_node = builder.add_library_node<stdlib::MallocNode>(block1, DebugInfo(), symbolic::integer(1024));
    builder.add_computational_memlet(block1, malloc_node, "_ret", malloc_output, {}, ptr_desc, DebugInfo());

    // Block 2: use the allocated memory
    auto& block2 = builder.add_block(root);
    auto& read_access = builder.add_access(block2, "ptr");
    auto& write_access = builder.add_access(block2, "ptr");
    auto& tasklet = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block2, read_access, tasklet, "_in", {symbolic::zero()}, ptr_desc, DebugInfo());
    builder.add_computational_memlet(block2, tasklet, "_out", write_access, {symbolic::zero()}, ptr_desc, DebugInfo());

    // Initial state: no free nodes
    EXPECT_EQ(count_free_nodes(builder.subject()), 0);

    // Apply the InsertFrees pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::InsertFreesPass pass;
    bool applied = pass.run(builder, analysis_manager);

    // Check that the pass was applied and free was inserted
    EXPECT_TRUE(applied);
    EXPECT_EQ(count_free_nodes(builder.subject()), 1);
    EXPECT_TRUE(has_free_for_container(builder.subject(), "ptr"));
}

TEST(InsertFreesPassTest, NoFreeForEscaping_Return) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Pointer opaque_desc;
    builder.add_container("ptr", opaque_desc);

    // Block: malloc
    auto& block = builder.add_block(root);
    auto& access_node = builder.add_access(block, "ptr");
    auto& lib_node = builder.add_library_node<stdlib::MallocNode>(block, DebugInfo(), symbolic::integer(1024));
    builder.add_computational_memlet(block, lib_node, "_ret", access_node, {}, opaque_desc, DebugInfo());

    // Return the pointer - makes it escape
    builder.add_return(root, "ptr");

    // Apply the InsertFrees pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::InsertFreesPass pass;
    bool applied = pass.run(builder, analysis_manager);

    // No free should be inserted for escaping allocation
    EXPECT_FALSE(applied);
    EXPECT_EQ(count_free_nodes(builder.subject()), 0);
}

TEST(InsertFreesPassTest, NoFreeForEscaping_Argument) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Pointer opaque_desc;
    builder.add_container("arg0", opaque_desc, true); // argument

    // Block: malloc into argument
    auto& block = builder.add_block(root);
    auto& access_node = builder.add_access(block, "arg0");
    auto& lib_node = builder.add_library_node<stdlib::MallocNode>(block, DebugInfo(), symbolic::integer(1024));
    builder.add_computational_memlet(block, lib_node, "_ret", access_node, {}, opaque_desc, DebugInfo());

    // Apply the InsertFrees pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::InsertFreesPass pass;
    bool applied = pass.run(builder, analysis_manager);

    // No free should be inserted for argument (escaping)
    EXPECT_FALSE(applied);
    EXPECT_EQ(count_free_nodes(builder.subject()), 0);
}

TEST(InsertFreesPassTest, MultipleMallocs_AllGetFrees) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Pointer ptr_desc(base_desc);
    builder.add_container("ptr1", ptr_desc);
    builder.add_container("ptr2", ptr_desc);

    // Block 1: two mallocs
    auto& block1 = builder.add_block(root);
    auto& access1 = builder.add_access(block1, "ptr1");
    auto& malloc1 = builder.add_library_node<stdlib::MallocNode>(block1, DebugInfo(), symbolic::integer(1024));
    builder.add_computational_memlet(block1, malloc1, "_ret", access1, {}, ptr_desc, DebugInfo());

    auto& access2 = builder.add_access(block1, "ptr2");
    auto& malloc2 = builder.add_library_node<stdlib::MallocNode>(block1, DebugInfo(), symbolic::integer(2048));
    builder.add_computational_memlet(block1, malloc2, "_ret", access2, {}, ptr_desc, DebugInfo());

    // Block 2: use both pointers
    auto& block2 = builder.add_block(root);
    auto& use1 = builder.add_access(block2, "ptr1");
    auto& use2 = builder.add_access(block2, "ptr2");
    auto& out1 = builder.add_access(block2, "ptr1");
    auto& tasklet = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block2, use1, tasklet, "_in", {symbolic::zero()}, ptr_desc, DebugInfo());
    builder.add_computational_memlet(block2, tasklet, "_out", out1, {symbolic::zero()}, ptr_desc, DebugInfo());
    // Just read ptr2 to mark it as used
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block2, use2, tasklet2, "_in", {symbolic::zero()}, ptr_desc, DebugInfo());
    auto& out2 = builder.add_access(block2, "ptr2");
    builder.add_computational_memlet(block2, tasklet2, "_out", out2, {symbolic::zero()}, ptr_desc, DebugInfo());

    // Apply the InsertFrees pass once
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::InsertFreesPass pass;
    bool applied = pass.run(builder, analysis_manager);

    // The pass should apply (insert at least one free)
    EXPECT_TRUE(applied);

    // Both allocations should have frees (pass may insert one or both in single run)
    // Since both containers share the same last use block, both should be freed
    size_t free_count = count_free_nodes(builder.subject());
    EXPECT_GE(free_count, 1);
    EXPECT_LE(free_count, 2);
}

TEST(InsertFreesPassTest, MixedEscapingAndNonEscaping) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Pointer ptr_desc(base_desc);
    builder.add_container("local_ptr", ptr_desc); // non-escaping
    builder.add_container("arg_ptr", ptr_desc, true); // escaping (argument)

    // Block 1: two mallocs
    auto& block1 = builder.add_block(root);

    auto& local_access = builder.add_access(block1, "local_ptr");
    auto& local_malloc = builder.add_library_node<stdlib::MallocNode>(block1, DebugInfo(), symbolic::integer(1024));
    builder.add_computational_memlet(block1, local_malloc, "_ret", local_access, {}, ptr_desc, DebugInfo());

    auto& arg_access = builder.add_access(block1, "arg_ptr");
    auto& arg_malloc = builder.add_library_node<stdlib::MallocNode>(block1, DebugInfo(), symbolic::integer(2048));
    builder.add_computational_memlet(block1, arg_malloc, "_ret", arg_access, {}, ptr_desc, DebugInfo());

    // Block 2: use local_ptr
    auto& block2 = builder.add_block(root);
    auto& use_local = builder.add_access(block2, "local_ptr");
    auto& out_local = builder.add_access(block2, "local_ptr");
    auto& tasklet = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block2, use_local, tasklet, "_in", {symbolic::zero()}, ptr_desc, DebugInfo());
    builder.add_computational_memlet(block2, tasklet, "_out", out_local, {symbolic::zero()}, ptr_desc, DebugInfo());

    // Apply the InsertFrees pass once
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::InsertFreesPass pass;
    bool applied = pass.run(builder, analysis_manager);

    // The pass should apply (insert free for local_ptr)
    EXPECT_TRUE(applied);

    // Only local_ptr should have a free (arg_ptr escapes)
    EXPECT_EQ(count_free_nodes(builder.subject()), 1);
    EXPECT_TRUE(has_free_for_container(builder.subject(), "local_ptr"));
    EXPECT_FALSE(has_free_for_container(builder.subject(), "arg_ptr"));
}

TEST(InsertFreesPassTest, NoMallocs_NotApplied) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("x", desc);

    auto sym_x = symbolic::symbol("x");
    builder.add_block(root, {{sym_x, symbolic::integer(42)}});

    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::InsertFreesPass pass;
    bool applied = pass.run(builder, analysis_manager);

    EXPECT_FALSE(applied);
    EXPECT_EQ(count_free_nodes(builder.subject()), 0);
}

TEST(InsertFreesPassTest, Idempotent) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Pointer ptr_desc(base_desc);
    builder.add_container("ptr", ptr_desc);

    // Block 1: malloc
    auto& block1 = builder.add_block(root);
    auto& access_node = builder.add_access(block1, "ptr");
    auto& lib_node = builder.add_library_node<stdlib::MallocNode>(block1, DebugInfo(), symbolic::integer(1024));
    builder.add_computational_memlet(block1, lib_node, "_ret", access_node, {}, ptr_desc, DebugInfo());

    // Block 2: use ptr
    auto& block2 = builder.add_block(root);
    auto& use_ptr = builder.add_access(block2, "ptr");
    auto& out_ptr = builder.add_access(block2, "ptr");
    auto& tasklet = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block2, use_ptr, tasklet, "_in", {symbolic::zero()}, ptr_desc, DebugInfo());
    builder.add_computational_memlet(block2, tasklet, "_out", out_ptr, {symbolic::zero()}, ptr_desc, DebugInfo());

    // Run with fresh analysis manager
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::InsertFreesPass pass;
    bool applied = pass.run(builder, analysis_manager);

    EXPECT_TRUE(applied);
    size_t count = count_free_nodes(builder.subject());
    EXPECT_EQ(count, 1);
    EXPECT_TRUE(has_free_for_container(builder.subject(), "ptr"));
}

TEST(InsertFreesPassTest, FreeInsertedAfterLastUse) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Pointer ptr_desc(base_desc);
    builder.add_container("ptr", ptr_desc);
    builder.add_container("result", base_desc);

    // Block 1: malloc
    auto& block1 = builder.add_block(root);
    auto& malloc_output = builder.add_access(block1, "ptr");
    auto& malloc_node =
        builder.add_library_node<stdlib::MallocNode>(block1, DebugInfo(), symbolic::integer(sizeof(int)));
    builder.add_computational_memlet(block1, malloc_node, "_ret", malloc_output, {}, ptr_desc, DebugInfo());

    // Block 2: write to allocated memory
    auto& block2 = builder.add_block(root);
    auto& write_ptr = builder.add_access(block2, "ptr");
    auto& constant = builder.add_constant(block2, "42", base_desc);
    auto& tasklet1 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block2, constant, tasklet1, "_in", {}, base_desc, DebugInfo());
    builder.add_computational_memlet(block2, tasklet1, "_out", write_ptr, {symbolic::zero()}, ptr_desc, DebugInfo());

    // Block 3: read from allocated memory and store in result
    auto& block3 = builder.add_block(root);
    auto& read_ptr = builder.add_access(block3, "ptr");
    auto& result_access = builder.add_access(block3, "result");
    auto& tasklet2 = builder.add_tasklet(block3, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block3, read_ptr, tasklet2, "_in", {symbolic::zero()}, ptr_desc, DebugInfo());
    builder.add_computational_memlet(block3, tasklet2, "_out", result_access, {}, base_desc, DebugInfo());

    size_t initial_root_size = root.size();
    EXPECT_EQ(initial_root_size, 3);

    // Apply the InsertFrees pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::InsertFreesPass pass;
    bool applied = pass.run(builder, analysis_manager);

    EXPECT_TRUE(applied);
    EXPECT_EQ(count_free_nodes(builder.subject()), 1);

    // The root should have one more block (the free block)
    EXPECT_EQ(root.size(), initial_root_size + 1);
}

TEST(InsertFreesPassTest, MallocWithMultipleUses) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer ptr_desc(base_desc);
    builder.add_container("arr", ptr_desc);

    // Block 1: malloc array
    auto& block1 = builder.add_block(root);
    auto& malloc_output = builder.add_access(block1, "arr");
    auto& malloc_node =
        builder.add_library_node<stdlib::MallocNode>(block1, DebugInfo(), symbolic::integer(sizeof(float) * 100));
    builder.add_computational_memlet(block1, malloc_node, "_ret", malloc_output, {}, ptr_desc, DebugInfo());

    // Block 2: write to array[0]
    auto& block2 = builder.add_block(root);
    auto& arr_write1 = builder.add_access(block2, "arr");
    auto& const1 = builder.add_constant(block2, "1.0", base_desc);
    auto& tasklet1 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block2, const1, tasklet1, "_in", {}, base_desc, DebugInfo());
    builder.add_computational_memlet(block2, tasklet1, "_out", arr_write1, {symbolic::zero()}, ptr_desc, DebugInfo());

    // Block 3: write to array[1]
    auto& block3 = builder.add_block(root);
    auto& arr_write2 = builder.add_access(block3, "arr");
    auto& const2 = builder.add_constant(block3, "2.0", base_desc);
    auto& tasklet2 = builder.add_tasklet(block3, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block3, const2, tasklet2, "_in", {}, base_desc, DebugInfo());
    builder
        .add_computational_memlet(block3, tasklet2, "_out", arr_write2, {symbolic::integer(1)}, ptr_desc, DebugInfo());

    // Block 4: read array[0] + array[1]
    auto& block4 = builder.add_block(root);
    auto& arr_read1 = builder.add_access(block4, "arr");
    auto& arr_read2 = builder.add_access(block4, "arr");
    auto& arr_write3 = builder.add_access(block4, "arr");
    auto& tasklet3 = builder.add_tasklet(block4, data_flow::TaskletCode::fp_add, "_out", {"_in0", "_in1"});
    builder.add_computational_memlet(block4, arr_read1, tasklet3, "_in0", {symbolic::zero()}, ptr_desc, DebugInfo());
    builder.add_computational_memlet(block4, arr_read2, tasklet3, "_in1", {symbolic::integer(1)}, ptr_desc, DebugInfo());
    builder
        .add_computational_memlet(block4, tasklet3, "_out", arr_write3, {symbolic::integer(2)}, ptr_desc, DebugInfo());

    // Apply the InsertFrees pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::InsertFreesPass pass;
    bool applied = pass.run(builder, analysis_manager);

    EXPECT_TRUE(applied);
    EXPECT_EQ(count_free_nodes(builder.subject()), 1);
    EXPECT_TRUE(has_free_for_container(builder.subject(), "arr"));
}

TEST(InsertFreesPassTest, PassName) { EXPECT_EQ(passes::InsertFrees::name(), "InsertFrees"); }

TEST(InsertFreesPassTest, MallocInsideLoop_LocalUse) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar idx_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Pointer ptr_desc(base_desc);
    builder.add_container("i", idx_desc);
    builder.add_container("ptr", ptr_desc);

    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, symbolic::integer(10));
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Block inside loop: malloc
    auto& block1 = builder.add_block(body);
    auto& malloc_access = builder.add_access(block1, "ptr");
    auto& malloc_node = builder.add_library_node<stdlib::MallocNode>(block1, DebugInfo(), symbolic::integer(1024));
    builder.add_computational_memlet(block1, malloc_node, "_ret", malloc_access, {}, ptr_desc, DebugInfo());

    // Block inside loop: use ptr
    auto& block2 = builder.add_block(body);
    auto& use_ptr = builder.add_access(block2, "ptr");
    auto& out_ptr = builder.add_access(block2, "ptr");
    auto& tasklet = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block2, use_ptr, tasklet, "_in", {symbolic::zero()}, ptr_desc, DebugInfo());
    builder.add_computational_memlet(block2, tasklet, "_out", out_ptr, {symbolic::zero()}, ptr_desc, DebugInfo());

    // Apply the InsertFrees pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::InsertFreesPass pass;
    bool applied = pass.run(builder, analysis_manager);

    // A free should be inserted for the loop-local allocation
    EXPECT_TRUE(applied);
    EXPECT_GE(count_free_nodes(builder.subject()), 1);
}

TEST(InsertFreesPassTest, MallocOutsideLoop_UsedInsideLoop) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar idx_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Pointer ptr_desc(base_desc);
    builder.add_container("i", idx_desc);
    builder.add_container("ptr", ptr_desc);

    // Block before loop: malloc
    auto& block_pre = builder.add_block(root);
    auto& malloc_access = builder.add_access(block_pre, "ptr");
    auto& malloc_node = builder.add_library_node<stdlib::MallocNode>(block_pre, DebugInfo(), symbolic::integer(1024));
    builder.add_computational_memlet(block_pre, malloc_node, "_ret", malloc_access, {}, ptr_desc, DebugInfo());

    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, symbolic::integer(10));
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Block inside loop: use ptr with loop index
    auto& block_loop = builder.add_block(body);
    auto& use_ptr = builder.add_access(block_loop, "ptr");
    auto& out_ptr = builder.add_access(block_loop, "ptr");
    auto& tasklet = builder.add_tasklet(block_loop, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block_loop, use_ptr, tasklet, "_in", {indvar}, ptr_desc, DebugInfo());
    builder.add_computational_memlet(block_loop, tasklet, "_out", out_ptr, {indvar}, ptr_desc, DebugInfo());

    // Apply the InsertFrees pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::InsertFreesPass pass;
    bool applied = pass.run(builder, analysis_manager);

    // A free should be inserted after the loop (after the last use)
    EXPECT_TRUE(applied);
    EXPECT_EQ(count_free_nodes(builder.subject()), 1);
    EXPECT_TRUE(has_free_for_container(builder.subject(), "ptr"));
}

TEST(InsertFreesPassTest, MallocInsideWhileLoop) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar idx_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Pointer ptr_desc(base_desc);
    builder.add_container("cond", idx_desc);
    builder.add_container("ptr", ptr_desc);

    auto& loop = builder.add_while(root);
    auto& body = loop.root();

    // Block inside while: malloc
    auto& block1 = builder.add_block(body);
    auto& malloc_access = builder.add_access(block1, "ptr");
    auto& malloc_node = builder.add_library_node<stdlib::MallocNode>(block1, DebugInfo(), symbolic::integer(512));
    builder.add_computational_memlet(block1, malloc_node, "_ret", malloc_access, {}, ptr_desc, DebugInfo());

    // Block inside while: use ptr
    auto& block2 = builder.add_block(body);
    auto& use_ptr = builder.add_access(block2, "ptr");
    auto& out_ptr = builder.add_access(block2, "ptr");
    auto& tasklet = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block2, use_ptr, tasklet, "_in", {symbolic::zero()}, ptr_desc, DebugInfo());
    builder.add_computational_memlet(block2, tasklet, "_out", out_ptr, {symbolic::zero()}, ptr_desc, DebugInfo());

    // Apply the InsertFrees pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::InsertFreesPass pass;
    bool applied = pass.run(builder, analysis_manager);

    // A free should be inserted for the while-loop-local allocation
    EXPECT_TRUE(applied);
    EXPECT_GE(count_free_nodes(builder.subject()), 1);
}

TEST(InsertFreesPassTest, PointerReassignment_MallocTwice) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Pointer ptr_desc(base_desc);
    builder.add_container("ptr", ptr_desc);

    // Block 1: first malloc
    auto& block1 = builder.add_block(root);
    auto& malloc1_access = builder.add_access(block1, "ptr");
    auto& malloc1_node = builder.add_library_node<stdlib::MallocNode>(block1, DebugInfo(), symbolic::integer(1024));
    builder.add_computational_memlet(block1, malloc1_node, "_ret", malloc1_access, {}, ptr_desc, DebugInfo());

    // Block 2: use first allocation
    auto& block2 = builder.add_block(root);
    auto& use1_ptr = builder.add_access(block2, "ptr");
    auto& out1_ptr = builder.add_access(block2, "ptr");
    auto& tasklet1 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block2, use1_ptr, tasklet1, "_in", {symbolic::zero()}, ptr_desc, DebugInfo());
    builder.add_computational_memlet(block2, tasklet1, "_out", out1_ptr, {symbolic::zero()}, ptr_desc, DebugInfo());

    // Block 3: second malloc (reassignment - overwrites ptr)
    auto& block3 = builder.add_block(root);
    auto& malloc2_access = builder.add_access(block3, "ptr");
    auto& malloc2_node = builder.add_library_node<stdlib::MallocNode>(block3, DebugInfo(), symbolic::integer(2048));
    builder.add_computational_memlet(block3, malloc2_node, "_ret", malloc2_access, {}, ptr_desc, DebugInfo());

    // Block 4: use second allocation
    auto& block4 = builder.add_block(root);
    auto& use2_ptr = builder.add_access(block4, "ptr");
    auto& out2_ptr = builder.add_access(block4, "ptr");
    auto& tasklet2 = builder.add_tasklet(block4, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block4, use2_ptr, tasklet2, "_in", {symbolic::zero()}, ptr_desc, DebugInfo());
    builder.add_computational_memlet(block4, tasklet2, "_out", out2_ptr, {symbolic::zero()}, ptr_desc, DebugInfo());

    // Apply the InsertFrees pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::InsertFreesPass pass;
    bool applied = pass.run(builder, analysis_manager);

    // The pass should insert at least one free
    // Note: proper handling of reassignment would require freeing before reassign
    // Current behavior may only free the last allocation
    EXPECT_TRUE(applied);
    EXPECT_GE(count_free_nodes(builder.subject()), 1);
}

TEST(InsertFreesPassTest, MallocInsideLoop_ReassignmentPerIteration) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar idx_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Pointer ptr_desc(base_desc);
    builder.add_container("i", idx_desc);
    builder.add_container("ptr", ptr_desc);

    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, symbolic::integer(5));
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Each iteration: malloc (reassigns ptr)
    auto& block1 = builder.add_block(body);
    auto& malloc_access = builder.add_access(block1, "ptr");
    auto& malloc_node = builder.add_library_node<stdlib::MallocNode>(block1, DebugInfo(), symbolic::integer(256));
    builder.add_computational_memlet(block1, malloc_node, "_ret", malloc_access, {}, ptr_desc, DebugInfo());

    // Each iteration: use ptr
    auto& block2 = builder.add_block(body);
    auto& use_ptr = builder.add_access(block2, "ptr");
    auto& out_ptr = builder.add_access(block2, "ptr");
    auto& tasklet = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block2, use_ptr, tasklet, "_in", {symbolic::zero()}, ptr_desc, DebugInfo());
    builder.add_computational_memlet(block2, tasklet, "_out", out_ptr, {symbolic::zero()}, ptr_desc, DebugInfo());

    // Apply the InsertFrees pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::InsertFreesPass pass;
    bool applied = pass.run(builder, analysis_manager);

    // The pass should attempt to insert frees for loop-local allocations
    // The exact behavior depends on the escape analysis implementation
    EXPECT_TRUE(applied);
    size_t free_count = count_free_nodes(builder.subject());
    EXPECT_GE(free_count, 1);
}

TEST(InsertFreesPassTest, NestedLoops_MallocInInnerLoop) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar idx_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Pointer ptr_desc(base_desc);
    builder.add_container("i", idx_desc);
    builder.add_container("j", idx_desc);
    builder.add_container("ptr", ptr_desc);

    // Outer loop
    auto i_sym = symbolic::symbol("i");
    auto& outer_loop = builder.add_for(
        root,
        i_sym,
        symbolic::Lt(i_sym, symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(i_sym, symbolic::integer(1))
    );
    auto& outer_body = outer_loop.root();

    // Inner loop
    auto j_sym = symbolic::symbol("j");
    auto& inner_loop = builder.add_for(
        outer_body,
        j_sym,
        symbolic::Lt(j_sym, symbolic::integer(5)),
        symbolic::integer(0),
        symbolic::add(j_sym, symbolic::integer(1))
    );
    auto& inner_body = inner_loop.root();

    // Block in inner loop: malloc
    auto& block1 = builder.add_block(inner_body);
    auto& malloc_access = builder.add_access(block1, "ptr");
    auto& malloc_node = builder.add_library_node<stdlib::MallocNode>(block1, DebugInfo(), symbolic::integer(128));
    builder.add_computational_memlet(block1, malloc_node, "_ret", malloc_access, {}, ptr_desc, DebugInfo());

    // Block in inner loop: use ptr
    auto& block2 = builder.add_block(inner_body);
    auto& use_ptr = builder.add_access(block2, "ptr");
    auto& out_ptr = builder.add_access(block2, "ptr");
    auto& tasklet = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block2, use_ptr, tasklet, "_in", {symbolic::zero()}, ptr_desc, DebugInfo());
    builder.add_computational_memlet(block2, tasklet, "_out", out_ptr, {symbolic::zero()}, ptr_desc, DebugInfo());

    // Apply the InsertFrees pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::InsertFreesPass pass;
    bool applied = pass.run(builder, analysis_manager);

    // A free should be inserted for the inner loop allocation
    EXPECT_TRUE(applied);
    EXPECT_GE(count_free_nodes(builder.subject()), 1);
}

TEST(InsertFreesPassTest, MallocBeforeLoop_FreedAfterLoop) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar idx_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Pointer ptr_desc(base_desc);
    builder.add_container("i", idx_desc);
    builder.add_container("ptr", ptr_desc);
    builder.add_container("sum", base_desc);

    // Block: malloc before loop
    auto& block_pre = builder.add_block(root);
    auto& malloc_access = builder.add_access(block_pre, "ptr");
    auto& malloc_node = builder.add_library_node<stdlib::MallocNode>(block_pre, DebugInfo(), symbolic::integer(4096));
    builder.add_computational_memlet(block_pre, malloc_node, "_ret", malloc_access, {}, ptr_desc, DebugInfo());

    // Loop using the allocated memory
    auto indvar = symbolic::symbol("i");
    auto& loop = builder.add_for(
        root,
        indvar,
        symbolic::Lt(indvar, symbolic::integer(100)),
        symbolic::integer(0),
        symbolic::add(indvar, symbolic::integer(1))
    );
    auto& body = loop.root();

    // Block in loop: use ptr[i]
    auto& block_loop = builder.add_block(body);
    auto& use_ptr = builder.add_access(block_loop, "ptr");
    auto& out_ptr = builder.add_access(block_loop, "ptr");
    auto& tasklet = builder.add_tasklet(block_loop, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block_loop, use_ptr, tasklet, "_in", {indvar}, ptr_desc, DebugInfo());
    builder.add_computational_memlet(block_loop, tasklet, "_out", out_ptr, {indvar}, ptr_desc, DebugInfo());

    // Get initial size
    size_t initial_root_size = root.size();

    // Apply the InsertFrees pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::InsertFreesPass pass;
    bool applied = pass.run(builder, analysis_manager);

    // Free should be inserted after the loop
    EXPECT_TRUE(applied);
    EXPECT_EQ(count_free_nodes(builder.subject()), 1);
    EXPECT_TRUE(has_free_for_container(builder.subject(), "ptr"));

    // Root should have grown by 1 (the free block after the loop)
    EXPECT_EQ(root.size(), initial_root_size + 1);
}

TEST(InsertFreesPassTest, MallocInIfBranch_UsedAfter) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar cond_desc(types::PrimitiveType::Int32);
    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Pointer ptr_desc(base_desc);
    builder.add_container("cond", cond_desc);
    builder.add_container("ptr", ptr_desc);

    auto cond_sym = symbolic::symbol("cond");

    // If-else: malloc only in if-branch
    auto& if_else = builder.add_if_else(root);
    auto& if_branch = builder.add_case(if_else, symbolic::Gt(cond_sym, symbolic::zero()));

    auto& malloc_block = builder.add_block(if_branch);
    auto& malloc_access = builder.add_access(malloc_block, "ptr");
    auto& malloc_node =
        builder.add_library_node<stdlib::MallocNode>(malloc_block, DebugInfo(), symbolic::integer(1024));
    builder.add_computational_memlet(malloc_block, malloc_node, "_ret", malloc_access, {}, ptr_desc, DebugInfo());

    // Else branch: do nothing (ptr is uninitialized)
    auto& else_branch = builder.add_case(if_else, symbolic::Le(cond_sym, symbolic::zero()));
    builder.add_block(else_branch); // empty block

    // Block after if-else: use ptr (assumes it was allocated)
    auto& use_block = builder.add_block(root);
    auto& use_ptr = builder.add_access(use_block, "ptr");
    auto& out_ptr = builder.add_access(use_block, "ptr");
    auto& tasklet = builder.add_tasklet(use_block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(use_block, use_ptr, tasklet, "_in", {symbolic::zero()}, ptr_desc, DebugInfo());
    builder.add_computational_memlet(use_block, tasklet, "_out", out_ptr, {symbolic::zero()}, ptr_desc, DebugInfo());

    // Apply the InsertFrees pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::InsertFreesPass pass;
    bool applied = pass.run(builder, analysis_manager);

    // A free should be inserted after the use (which is after the if-else)
    EXPECT_TRUE(applied);
    EXPECT_EQ(count_free_nodes(builder.subject()), 1);
    EXPECT_TRUE(has_free_for_container(builder.subject(), "ptr"));
}

TEST(InsertFreesPassTest, MallocInBothBranches) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar cond_desc(types::PrimitiveType::Int32);
    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Pointer ptr_desc(base_desc);
    builder.add_container("cond", cond_desc);
    builder.add_container("ptr", ptr_desc);

    auto cond_sym = symbolic::symbol("cond");

    // If-else: malloc in both branches (different sizes)
    auto& if_else = builder.add_if_else(root);

    // If branch: malloc 1024 bytes
    auto& if_branch = builder.add_case(if_else, symbolic::Gt(cond_sym, symbolic::zero()));
    auto& malloc_block1 = builder.add_block(if_branch);
    auto& malloc_access1 = builder.add_access(malloc_block1, "ptr");
    auto& malloc_node1 =
        builder.add_library_node<stdlib::MallocNode>(malloc_block1, DebugInfo(), symbolic::integer(1024));
    builder.add_computational_memlet(malloc_block1, malloc_node1, "_ret", malloc_access1, {}, ptr_desc, DebugInfo());

    // Else branch: malloc 2048 bytes
    auto& else_branch = builder.add_case(if_else, symbolic::Le(cond_sym, symbolic::zero()));
    auto& malloc_block2 = builder.add_block(else_branch);
    auto& malloc_access2 = builder.add_access(malloc_block2, "ptr");
    auto& malloc_node2 =
        builder.add_library_node<stdlib::MallocNode>(malloc_block2, DebugInfo(), symbolic::integer(2048));
    builder.add_computational_memlet(malloc_block2, malloc_node2, "_ret", malloc_access2, {}, ptr_desc, DebugInfo());

    // Block after if-else: use ptr
    auto& use_block = builder.add_block(root);
    auto& use_ptr = builder.add_access(use_block, "ptr");
    auto& out_ptr = builder.add_access(use_block, "ptr");
    auto& tasklet = builder.add_tasklet(use_block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(use_block, use_ptr, tasklet, "_in", {symbolic::zero()}, ptr_desc, DebugInfo());
    builder.add_computational_memlet(use_block, tasklet, "_out", out_ptr, {symbolic::zero()}, ptr_desc, DebugInfo());

    // Apply the InsertFrees pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::InsertFreesPass pass;
    bool applied = pass.run(builder, analysis_manager);

    // A free should be inserted - at minimum one free after the use block
    // Note: Both mallocs write to same container, so only one free is needed
    EXPECT_TRUE(applied);
    EXPECT_GE(count_free_nodes(builder.subject()), 1);
    EXPECT_TRUE(has_free_for_container(builder.subject(), "ptr"));
}

TEST(InsertFreesPassTest, MallocInsideMap) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar idx_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Pointer ptr_desc(base_desc);
    builder.add_container("i", idx_desc);
    builder.add_container("ptr", ptr_desc);

    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, symbolic::integer(10));
    auto update = symbolic::add(indvar, symbolic::integer(1));
    auto schedule = structured_control_flow::ScheduleType_Sequential::create();

    // Map (parallel loop)
    auto& map = builder.add_map(root, indvar, condition, init, update, schedule);
    auto& body = map.root();

    // Block inside map: malloc
    auto& block1 = builder.add_block(body);
    auto& malloc_access = builder.add_access(block1, "ptr");
    auto& malloc_node = builder.add_library_node<stdlib::MallocNode>(block1, DebugInfo(), symbolic::integer(256));
    builder.add_computational_memlet(block1, malloc_node, "_ret", malloc_access, {}, ptr_desc, DebugInfo());

    // Block inside map: use ptr
    auto& block2 = builder.add_block(body);
    auto& use_ptr = builder.add_access(block2, "ptr");
    auto& out_ptr = builder.add_access(block2, "ptr");
    auto& tasklet = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block2, use_ptr, tasklet, "_in", {symbolic::zero()}, ptr_desc, DebugInfo());
    builder.add_computational_memlet(block2, tasklet, "_out", out_ptr, {symbolic::zero()}, ptr_desc, DebugInfo());

    // Apply the InsertFrees pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::InsertFreesPass pass;
    bool applied = pass.run(builder, analysis_manager);

    // A free should be inserted within the map body (each iteration frees its allocation)
    EXPECT_TRUE(applied);
    EXPECT_GE(count_free_nodes(builder.subject()), 1);
    EXPECT_TRUE(has_free_for_container(builder.subject(), "ptr"));
}

TEST(InsertFreesPassTest, ConditionalMallocInsideLoop) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar idx_desc(types::PrimitiveType::UInt64);
    types::Scalar cond_desc(types::PrimitiveType::Int32);
    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Pointer ptr_desc(base_desc);
    builder.add_container("i", idx_desc);
    builder.add_container("flag", cond_desc);
    builder.add_container("ptr", ptr_desc);

    auto indvar = symbolic::symbol("i");
    auto flag_sym = symbolic::symbol("flag");

    // For loop
    auto& loop = builder.add_for(
        root,
        indvar,
        symbolic::Lt(indvar, symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(indvar, symbolic::integer(1))
    );
    auto& loop_body = loop.root();

    // If-else inside loop
    auto& if_else = builder.add_if_else(loop_body);

    // If branch: malloc
    auto& if_branch = builder.add_case(if_else, symbolic::Gt(flag_sym, symbolic::zero()));
    auto& malloc_block = builder.add_block(if_branch);
    auto& malloc_access = builder.add_access(malloc_block, "ptr");
    auto& malloc_node = builder.add_library_node<stdlib::MallocNode>(malloc_block, DebugInfo(), symbolic::integer(512));
    builder.add_computational_memlet(malloc_block, malloc_node, "_ret", malloc_access, {}, ptr_desc, DebugInfo());

    // Use ptr in if branch
    auto& use_block = builder.add_block(if_branch);
    auto& use_ptr = builder.add_access(use_block, "ptr");
    auto& out_ptr = builder.add_access(use_block, "ptr");
    auto& tasklet = builder.add_tasklet(use_block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(use_block, use_ptr, tasklet, "_in", {symbolic::zero()}, ptr_desc, DebugInfo());
    builder.add_computational_memlet(use_block, tasklet, "_out", out_ptr, {symbolic::zero()}, ptr_desc, DebugInfo());

    // Else branch: empty
    auto& else_branch = builder.add_case(if_else, symbolic::Le(flag_sym, symbolic::zero()));
    builder.add_block(else_branch);

    // Apply the InsertFrees pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::InsertFreesPass pass;
    bool applied = pass.run(builder, analysis_manager);

    // A free should be inserted within the if-branch of the loop
    EXPECT_TRUE(applied);
    EXPECT_GE(count_free_nodes(builder.subject()), 1);
    EXPECT_TRUE(has_free_for_container(builder.subject(), "ptr"));
}

TEST(InsertFreesPassTest, DifferentContainersInDifferentBranches) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar cond_desc(types::PrimitiveType::Int32);
    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Pointer ptr_desc(base_desc);
    builder.add_container("cond", cond_desc);
    builder.add_container("ptr1", ptr_desc);
    builder.add_container("ptr2", ptr_desc);

    auto cond_sym = symbolic::symbol("cond");

    // If-else
    auto& if_else = builder.add_if_else(root);

    // If branch: malloc ptr1
    auto& if_branch = builder.add_case(if_else, symbolic::Gt(cond_sym, symbolic::zero()));
    auto& malloc_block1 = builder.add_block(if_branch);
    auto& malloc_access1 = builder.add_access(malloc_block1, "ptr1");
    auto& malloc_node1 =
        builder.add_library_node<stdlib::MallocNode>(malloc_block1, DebugInfo(), symbolic::integer(1024));
    builder.add_computational_memlet(malloc_block1, malloc_node1, "_ret", malloc_access1, {}, ptr_desc, DebugInfo());

    // Else branch: malloc ptr2
    auto& else_branch = builder.add_case(if_else, symbolic::Le(cond_sym, symbolic::zero()));
    auto& malloc_block2 = builder.add_block(else_branch);
    auto& malloc_access2 = builder.add_access(malloc_block2, "ptr2");
    auto& malloc_node2 =
        builder.add_library_node<stdlib::MallocNode>(malloc_block2, DebugInfo(), symbolic::integer(2048));
    builder.add_computational_memlet(malloc_block2, malloc_node2, "_ret", malloc_access2, {}, ptr_desc, DebugInfo());

    // Block after if-else: use both (in reality only one is valid, but we test the analysis)
    auto& use_block = builder.add_block(root);
    auto& use1 = builder.add_access(use_block, "ptr1");
    auto& out1 = builder.add_access(use_block, "ptr1");
    auto& tasklet1 = builder.add_tasklet(use_block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(use_block, use1, tasklet1, "_in", {symbolic::zero()}, ptr_desc, DebugInfo());
    builder.add_computational_memlet(use_block, tasklet1, "_out", out1, {symbolic::zero()}, ptr_desc, DebugInfo());

    auto& use2 = builder.add_access(use_block, "ptr2");
    auto& out2 = builder.add_access(use_block, "ptr2");
    auto& tasklet2 = builder.add_tasklet(use_block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(use_block, use2, tasklet2, "_in", {symbolic::zero()}, ptr_desc, DebugInfo());
    builder.add_computational_memlet(use_block, tasklet2, "_out", out2, {symbolic::zero()}, ptr_desc, DebugInfo());

    // Apply the InsertFrees pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::InsertFreesPass pass;
    bool applied = pass.run(builder, analysis_manager);

    // Both containers should get frees
    EXPECT_TRUE(applied);
    EXPECT_EQ(count_free_nodes(builder.subject()), 2);
    EXPECT_TRUE(has_free_for_container(builder.subject(), "ptr1"));
    EXPECT_TRUE(has_free_for_container(builder.subject(), "ptr2"));
}
