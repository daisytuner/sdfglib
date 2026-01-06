#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/while.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"

using namespace sdfg;

namespace sdfg::structured_control_flow {

// Test For loop structure and pointers
TEST(ForLoopTest, BasicStructure) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);
    
    auto& root = builder.subject().root();
    
    auto& for_loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );
    
    // Verify for_loop is a StructuredLoop
    EXPECT_TRUE(dynamic_cast<const StructuredLoop*>(&for_loop) != nullptr);
    
    // Verify for_loop is a For
    EXPECT_TRUE(dynamic_cast<const For*>(&for_loop) != nullptr);
}

// Test For loop parameters
TEST(ForLoopTest, LoopParameters) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);
    
    auto& root = builder.subject().root();
    
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto update = symbolic::add(symbolic::symbol("i"), symbolic::integer(1));
    auto condition = symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10));
    
    auto& for_loop = builder.add_for(root, indvar, condition, init, update);
    
    // Verify parameters
    EXPECT_TRUE(symbolic::eq(for_loop.indvar(), indvar));
    EXPECT_TRUE(symbolic::eq(for_loop.init(), init));
    EXPECT_TRUE(symbolic::eq(for_loop.update(), update));
    EXPECT_TRUE(symbolic::eq(for_loop.condition(), condition));
}

// Test For loop root sequence
TEST(ForLoopTest, RootSequence) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);
    
    auto& root = builder.subject().root();
    
    auto& for_loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );
    
    // Verify loop has a root sequence
    auto& loop_root = for_loop.root();
    EXPECT_TRUE(dynamic_cast<const Sequence*>(&loop_root) != nullptr);
    
    // Add block to loop body
    builder.add_block(loop_root, control_flow::Assignments{});
    EXPECT_EQ(loop_root.size(), 1);
}

// Test While loop structure and pointers
TEST(WhileLoopTest, BasicStructure) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("x", int_type);
    
    auto& root = builder.subject().root();
    
    auto& while_loop = builder.add_while(root);
    
    // Verify while_loop is a ControlFlowNode
    EXPECT_TRUE(dynamic_cast<const ControlFlowNode*>(&while_loop) != nullptr);
    
    // Verify while_loop is a While
    EXPECT_TRUE(dynamic_cast<const While*>(&while_loop) != nullptr);
}

// Test While loop root sequence
TEST(WhileLoopTest, RootSequence) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    
    auto& root = builder.subject().root();
    auto& while_loop = builder.add_while(root);
    
    // Verify loop has a root sequence
    auto& loop_root = while_loop.root();
    EXPECT_TRUE(dynamic_cast<const Sequence*>(&loop_root) != nullptr);
    
    // Add block to loop body
    builder.add_block(loop_root, control_flow::Assignments{});
    EXPECT_EQ(loop_root.size(), 1);
    
    // Const access
    const auto& const_while = while_loop;
    const auto& const_root = const_while.root();
    EXPECT_TRUE(dynamic_cast<const Sequence*>(&const_root) != nullptr);
}

// Test Map structure and pointers
TEST(MapTest, BasicStructure) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);
    
    auto& root = builder.subject().root();
    
    auto& map = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        ScheduleType_Sequential::create()
    );
    
    // Verify map is a StructuredLoop
    EXPECT_TRUE(dynamic_cast<const StructuredLoop*>(&map) != nullptr);
    
    // Verify map is a Map
    EXPECT_TRUE(dynamic_cast<const Map*>(&map) != nullptr);
}

// Test Map parameters
TEST(MapTest, LoopParameters) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);
    
    auto& root = builder.subject().root();
    
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto update = symbolic::add(symbolic::symbol("i"), symbolic::integer(1));
    auto condition = symbolic::Lt(symbolic::symbol("i"), symbolic::integer(100));
    
    auto& map = builder.add_map(root, indvar, condition, init, update, ScheduleType_Sequential::create());
    
    // Verify parameters
    EXPECT_TRUE(symbolic::eq(map.indvar(), indvar));
    EXPECT_TRUE(symbolic::eq(map.init(), init));
    EXPECT_TRUE(symbolic::eq(map.update(), update));
    EXPECT_TRUE(symbolic::eq(map.condition(), condition));
}

// Test Map schedule type
TEST(MapTest, ScheduleType) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);
    
    auto& root = builder.subject().root();
    
    // Test Sequential schedule
    auto schedule_seq = ScheduleType_Sequential::create();
    auto& map_seq = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        schedule_seq
    );
    
    EXPECT_EQ(map_seq.schedule_type().value(), ScheduleType_Sequential::value());
    
    // Test CPU Parallel schedule
    auto schedule_par = ScheduleType_CPU_Parallel::create();
    auto& map_par = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        schedule_par
    );
    
    EXPECT_EQ(map_par.schedule_type().value(), ScheduleType_CPU_Parallel::value());
}

// Test Break and Continue structures
TEST(BreakContinueTest, BasicStructure) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);
    
    auto& root = builder.subject().root();
    
    auto& while_loop = builder.add_while(root);
    auto& loop_root = while_loop.root();
    
    // Add break
    auto& break_node = builder.add_break(loop_root);
    EXPECT_TRUE(dynamic_cast<const Break*>(&break_node) != nullptr);
    EXPECT_TRUE(dynamic_cast<const ControlFlowNode*>(&break_node) != nullptr);
    
    // Add continue
    auto& continue_node = builder.add_continue(loop_root);
    EXPECT_TRUE(dynamic_cast<const Continue*>(&continue_node) != nullptr);
    EXPECT_TRUE(dynamic_cast<const ControlFlowNode*>(&continue_node) != nullptr);
}

// Test nested loops
TEST(LoopTest, NestedLoops) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);
    builder.add_container("j", int_type);
    
    auto& root = builder.subject().root();
    
    // Outer loop
    auto& outer_loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );
    
    auto& outer_root = outer_loop.root();
    
    // Inner loop
    auto& inner_loop = builder.add_for(
        outer_root,
        symbolic::symbol("j"),
        symbolic::Lt(symbolic::symbol("j"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1))
    );
    
    // Verify nesting
    EXPECT_EQ(root.size(), 1);
    EXPECT_EQ(outer_root.size(), 1);
    
    auto& inner_root = inner_loop.root();
    builder.add_block(inner_root, control_flow::Assignments{});
    EXPECT_EQ(inner_root.size(), 1);
}

} // namespace sdfg::structured_control_flow
