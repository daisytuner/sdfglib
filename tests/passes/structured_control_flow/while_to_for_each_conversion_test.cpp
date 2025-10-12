#include "sdfg/passes/structured_control_flow/while_to_for_each_conversion.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/symbolic/symbolic.h"

using namespace sdfg;

TEST(WhileToForEachConversionTest, LinkedList) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Pointer opaque_desc;
    builder.add_container("list", opaque_desc, true);
    builder.add_container("iter", opaque_desc);

    auto sym_iter = symbolic::symbol("iter");
    auto sym_list = symbolic::symbol("list");
    auto sym_nullptr = symbolic::__nullptr__();

    // Reinterpret cast pointers to pointer(pointer) for dereferencing
    types::Pointer ptr_ptr_desc(static_cast<const types::IType&>(opaque_desc));

    // Init: iter = *list
    {
        auto& block = builder.add_block(root);
        auto& list = builder.add_access(block, "list");
        auto& iter = builder.add_access(block, "iter");
        builder.add_dereference_memlet(block, list, iter, true, ptr_ptr_desc);
    }

    auto& loop = builder.add_while(root);
    auto& body = loop.root();

    // Update: iter = *iter
    auto& block = builder.add_block(body);
    auto& iter_in = builder.add_access(block, "iter");
    auto& iter_out = builder.add_access(block, "iter");
    builder.add_dereference_memlet(block, iter_in, iter_out, true, ptr_ptr_desc);

    // Condition: iter != nullptr -> continue
    auto& ifelse = builder.add_if_else(body);
    auto& continue_scope = builder.add_case(ifelse, symbolic::Ne(sym_iter, sym_nullptr));
    builder.add_continue(continue_scope);
    auto& break_scope = builder.add_case(ifelse, symbolic::Eq(sym_iter, sym_nullptr));
    builder.add_break(break_scope);

    // Analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::WhileToForEachConversion conversion_pass;
    EXPECT_TRUE(conversion_pass.run(builder, analysis_manager));

    // Check
    auto for_each_node = dynamic_cast<const structured_control_flow::ForEach*>(&builder.subject().root().at(1).first);
    EXPECT_TRUE(for_each_node != nullptr);
    EXPECT_FALSE(for_each_node->has_init());
    EXPECT_TRUE(symbolic::eq(for_each_node->iterator(), sym_iter));
    EXPECT_TRUE(symbolic::eq(for_each_node->end(), sym_nullptr));
    EXPECT_TRUE(symbolic::eq(for_each_node->update(), sym_iter));
}

TEST(WhileToForEachConversionTest, DoubleLinkedList) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Pointer opaque_desc;
    builder.add_container("list", opaque_desc, true);
    builder.add_container("iter", opaque_desc);

    auto sym_iter = symbolic::symbol("iter");
    auto sym_list = symbolic::symbol("list");

    // Reinterpret cast pointers to pointer(pointer) for dereferencing
    types::Pointer ptr_ptr_desc(static_cast<const types::IType&>(opaque_desc));

    // Init: iter = *list
    {
        auto& block = builder.add_block(root);
        auto& list = builder.add_access(block, "list");
        auto& iter = builder.add_access(block, "iter");
        builder.add_dereference_memlet(block, list, iter, true, ptr_ptr_desc);
    }

    auto& loop = builder.add_while(root);
    auto& body = loop.root();

    // Update: iter = *iter
    auto& block = builder.add_block(body);
    auto& iter_in = builder.add_access(block, "iter");
    auto& iter_out = builder.add_access(block, "iter");
    builder.add_dereference_memlet(block, iter_in, iter_out, true, ptr_ptr_desc);

    // Condition: iter != nullptr -> continue
    auto& ifelse = builder.add_if_else(body);
    auto& continue_scope = builder.add_case(ifelse, symbolic::Ne(sym_iter, sym_list));
    builder.add_continue(continue_scope);
    auto& break_scope = builder.add_case(ifelse, symbolic::Eq(sym_iter, sym_list));
    builder.add_break(break_scope);

    // Analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::WhileToForEachConversion conversion_pass;
    EXPECT_TRUE(conversion_pass.run(builder, analysis_manager));

    // Check
    auto for_each_node = dynamic_cast<const structured_control_flow::ForEach*>(&builder.subject().root().at(1).first);
    EXPECT_TRUE(for_each_node != nullptr);
    EXPECT_FALSE(for_each_node->has_init());
    EXPECT_TRUE(symbolic::eq(for_each_node->iterator(), sym_iter));
    EXPECT_TRUE(symbolic::eq(for_each_node->end(), sym_list));
    EXPECT_TRUE(symbolic::eq(for_each_node->update(), sym_iter));
}

TEST(WhileToForEachConversionTest, DoubleLinkedList_NextPtrWithOffset) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Pointer opaque_desc;
    builder.add_container("list", opaque_desc, true);
    builder.add_container("iter", opaque_desc);
    builder.add_container("next_ptr", opaque_desc);

    auto sym_iter = symbolic::symbol("iter");
    auto sym_list = symbolic::symbol("list");
    auto sym_next_ptr = symbolic::symbol("next_ptr");

    // Reinterpret cast pointers to pointer(pointer) for dereferencing
    types::Pointer ptr_ptr_desc(static_cast<const types::IType&>(opaque_desc));

    // Init
    {
        // next_ptr = list + offset
        auto& block1 = builder.add_block(root);
        auto& list = builder.add_access(block1, "list");
        auto& next_ptr = builder.add_access(block1, "next_ptr");
        builder.add_reference_memlet(block1, list, next_ptr, {symbolic::integer(4)}, ptr_ptr_desc);

        auto& block2 = builder.add_block(root);
        auto& next_ptr2 = builder.add_access(block2, "next_ptr");
        auto& iter = builder.add_access(block2, "iter");
        builder.add_dereference_memlet(block2, next_ptr2, iter, true, ptr_ptr_desc);

    }

    auto& loop = builder.add_while(root);
    auto& body = loop.root();

    // Update
    {
        // next_ptr = iter + offset
        auto& block1 = builder.add_block(body);
        auto& iter_in = builder.add_access(block1, "iter");
        auto& next_ptr = builder.add_access(block1, "next_ptr");
        builder.add_reference_memlet(block1, iter_in, next_ptr, {symbolic::integer(4)}, ptr_ptr_desc);

        // iter = *next_ptr
        auto& block2 = builder.add_block(body);
        auto& next_ptr2 = builder.add_access(block2, "next_ptr");
        auto& iter_out = builder.add_access(block2, "iter");
        builder.add_dereference_memlet(block2, next_ptr2, iter_out, true, ptr_ptr_desc);
    }

    // Condition: iter != nullptr -> continue
    auto& ifelse = builder.add_if_else(body);
    auto& continue_scope = builder.add_case(ifelse, symbolic::Ne(sym_iter, sym_list));
    builder.add_continue(continue_scope);
    auto& break_scope = builder.add_case(ifelse, symbolic::Eq(sym_iter, sym_list));
    builder.add_break(break_scope);

    // Analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::WhileToForEachConversion conversion_pass;
    EXPECT_TRUE(conversion_pass.run(builder, analysis_manager));

    // Check
    auto for_each_node = dynamic_cast<const structured_control_flow::ForEach*>(&builder.subject().root().at(2).first);
    EXPECT_TRUE(for_each_node != nullptr);
    EXPECT_FALSE(for_each_node->has_init());
    EXPECT_TRUE(symbolic::eq(for_each_node->iterator(), sym_iter));
    EXPECT_TRUE(symbolic::eq(for_each_node->end(), sym_list));
    EXPECT_TRUE(symbolic::eq(for_each_node->update(), sym_next_ptr));
}
