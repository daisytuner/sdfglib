#include "sdfg/analysis/mem_access_range_analysis.h"

#include <gtest/gtest.h>
#include <iostream>
#include <symengine/symengine_rcp.h>


#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/type.h"
#include "sdfg/visualizer/dot_visualizer.h"

using namespace sdfg;

#ifndef DEBUG_WRITE_SDFG_VIZ
#define DEBUG_WRITE_SDFG_VIZ true
#endif

#define DEBUG_DOT_SDFG(sdfg)              \
    if constexpr (DEBUG_WRITE_SDFG_VIZ) { \
        writeSdfgDot(sdfg);               \
    }

static void writeSdfgDot(const StructuredSDFG& sdfg) {
    visualizer::DotVisualizer viz(sdfg);
    viz.visualize();

    std::string filename = sdfg.name() + ".dot";

    std::ofstream dotOutput(filename, std::ofstream::out);

    dotOutput << viz.getStream().str();
    dotOutput.close();
    std::cout << "Wrote graph to : " << filename << std::endl;
}

TEST(MemAccessRangeAnalysisTest, AccessNode_Write_Element_1D) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Pointer ptr_desc(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("i", base_desc, true);

    auto sym = symbolic::symbol("i");

    auto& root = builder.subject().root();

    auto& block = builder.add_block(root);

    auto& writeAccess = builder.add_access(block, "A");
    auto& zero_node = builder.add_constant(block, "0", base_desc);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, zero_node, tasklet, "_in", {});
    builder.add_computational_memlet(block, tasklet, "_out", writeAccess, {sym}, ptr_desc);

    auto sdfg = builder.move();

    DEBUG_DOT_SDFG(*sdfg);

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& ranges = analysis_manager.get<analysis::MemAccessRanges>();


    // Check result
    auto* range_a = ranges.get("A");
    EXPECT_NE(range_a, nullptr);
    EXPECT_EQ(range_a->get_name(), "A");
    EXPECT_FALSE(range_a->saw_read());
    EXPECT_TRUE(range_a->saw_write());
    EXPECT_FALSE(range_a->is_undefined());

    auto& dims = range_a->dims();
    EXPECT_EQ(dims.size(), 1);
    EXPECT_TRUE(symbolic::eq(dims[0].first, sym));
    EXPECT_TRUE(symbolic::eq(dims[0].second, sym));
}

TEST(MemAccessRangeAnalysisTest, AccessNode_Read_Element_1D) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Pointer ptr_desc(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("B", opaque_desc, true);
    builder.add_container("i", base_desc, true);

    auto sym = symbolic::symbol("i");

    auto& root = builder.subject().root();

    auto& block = builder.add_block(root);

    auto& node_A = builder.add_access(block, "A");
    auto& node_B = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, node_A, tasklet, "_in", {sym}, ptr_desc);
    builder.add_computational_memlet(block, tasklet, "_out", node_B, {sym}, ptr_desc);

    auto sdfg = builder.move();

    DEBUG_DOT_SDFG(*sdfg);

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& ranges = analysis_manager.get<analysis::MemAccessRanges>();


    // Check result
    auto* range_a = ranges.get("A");
    EXPECT_NE(range_a, nullptr);
    EXPECT_EQ(range_a->get_name(), "A");
    EXPECT_TRUE(range_a->saw_read());
    EXPECT_FALSE(range_a->saw_write());
    EXPECT_FALSE(range_a->is_undefined());

    auto& dims = range_a->dims();
    EXPECT_EQ(dims.size(), 1);
    EXPECT_TRUE(symbolic::eq(dims[0].first, sym));
    EXPECT_TRUE(symbolic::eq(dims[0].second, sym));
}

TEST(MemAccessRangeAnalysisTest, AccessNode_Write_Range_1D) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Pointer ptr_desc(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("N", base_desc, true);
    builder.add_container("i", base_desc);

    auto sym = symbolic::symbol("i");

    auto& root = builder.subject().root();
    auto& scope = builder.add_map(
        root,
        sym,
        symbolic::Lt(sym, symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(sym, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    auto& block = builder.add_block(scope.root());

    auto& node_A = builder.add_access(block, "A");
    auto& zero_node = builder.add_constant(block, "0", base_desc);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, zero_node, tasklet, "_in", {});
    builder.add_computational_memlet(block, tasklet, "_out", node_A, {sym}, ptr_desc);

    auto sdfg = builder.move();

    DEBUG_DOT_SDFG(*sdfg);

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& ranges = analysis_manager.get<analysis::MemAccessRanges>();

    // Check result
    auto* range_a = ranges.get("A");
    EXPECT_NE(range_a, nullptr);
    EXPECT_EQ(range_a->get_name(), "A");
    EXPECT_FALSE(range_a->saw_read());
    EXPECT_TRUE(range_a->saw_write());
    EXPECT_FALSE(range_a->is_undefined());

    auto& dims = range_a->dims();
    EXPECT_EQ(dims.size(), 1);
    EXPECT_TRUE(symbolic::eq(dims[0].first, symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(dims[0].second, symbolic::sub(symbolic::symbol("N"), symbolic::one())));
}

TEST(MemAccessRangeAnalysisTest, AccessNode_Write_Range_Shift_1D) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Pointer ptr_desc(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("N", base_desc, true);
    builder.add_container("i", base_desc);

    auto sym = symbolic::symbol("i");

    auto& root = builder.subject().root();
    auto& scope = builder.add_map(
        root,
        sym,
        symbolic::Lt(sym, symbolic::symbol("N")),
        symbolic::integer(10),
        symbolic::add(sym, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    auto& block = builder.add_block(scope.root());

    auto& node_A = builder.add_access(block, "A");
    auto& zero_node = builder.add_constant(block, "0", base_desc);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, zero_node, tasklet, "_in", {});
    builder.add_computational_memlet(block, tasklet, "_out", node_A, {sym}, ptr_desc);

    auto sdfg = builder.move();

    DEBUG_DOT_SDFG(*sdfg);

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& ranges = analysis_manager.get<analysis::MemAccessRanges>();

    // Check result
    auto* range_a = ranges.get("A");
    EXPECT_NE(range_a, nullptr);
    EXPECT_EQ(range_a->get_name(), "A");
    EXPECT_FALSE(range_a->saw_read());
    EXPECT_TRUE(range_a->saw_write());
    EXPECT_FALSE(range_a->is_undefined());

    auto& dims = range_a->dims();
    EXPECT_EQ(dims.size(), 1);
    EXPECT_TRUE(symbolic::eq(dims[0].first, symbolic::integer(10)));
    EXPECT_TRUE(symbolic::eq(dims[0].second, symbolic::sub(symbolic::symbol("N"), symbolic::one())));
}

TEST(MemAccessRangeAnalysisTest, AccessNode_Read_Range_1D) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Pointer ptr_desc(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("B", opaque_desc, true);
    builder.add_container("N", base_desc, true);
    builder.add_container("i", base_desc);

    auto sym = symbolic::symbol("i");

    auto& root = builder.subject().root();
    auto& scope = builder.add_map(
        root,
        sym,
        symbolic::Lt(sym, symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(sym, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    auto& block = builder.add_block(scope.root());

    auto& node_A = builder.add_access(block, "A");
    auto& node_B = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, node_A, tasklet, "_in", {sym}, ptr_desc);
    builder.add_computational_memlet(block, tasklet, "_out", node_B, {sym}, ptr_desc);

    auto sdfg = builder.move();

    DEBUG_DOT_SDFG(*sdfg);

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& ranges = analysis_manager.get<analysis::MemAccessRanges>();

    // Check result
    auto* range_a = ranges.get("A");
    EXPECT_NE(range_a, nullptr);
    EXPECT_EQ(range_a->get_name(), "A");
    EXPECT_TRUE(range_a->saw_read());
    EXPECT_FALSE(range_a->saw_write());
    EXPECT_FALSE(range_a->is_undefined());

    auto& dims = range_a->dims();
    EXPECT_EQ(dims.size(), 1);
    EXPECT_TRUE(symbolic::eq(dims[0].first, symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(dims[0].second, symbolic::sub(symbolic::symbol("N"), symbolic::one())));
}

TEST(MemAccessRangeAnalysisTest, AccessNode_Write_Range_2D) {
    builder::StructuredSDFGBuilder builder("sdfg_simple_2d", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Array array1dType(base_desc, symbolic::symbol("M"));
    types::Pointer array2dType(array1dType);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("arg_init", base_desc, true);
    builder.add_container("i", base_desc);
    builder.add_container("j", base_desc);
    auto sym_i = symbolic::symbol("i");
    auto sym_j = symbolic::symbol("j");

    auto& root = builder.subject().root();
    auto& outer_for = builder.add_for(
        root, sym_i, symbolic::Lt(sym_i, symbolic::integer(23)), symbolic::zero(), symbolic::add(symbolic::one(), sym_i)
    );
    auto& inner_for = builder.add_for(
        outer_for.root(),
        sym_j,
        symbolic::Lt(sym_j, symbolic::integer(16)),
        symbolic::zero(),
        symbolic::add(symbolic::one(), sym_j)
    );

    auto& block = builder.add_block(inner_for.root());
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    auto& readAccess = builder.add_access(block, "arg_init");
    auto& readArg = builder.add_computational_memlet(block, readAccess, tasklet, "_in", {});
    auto& writeAccess = builder.add_access(block, "A");
    auto& writeArg = builder.add_computational_memlet(block, tasklet, "_out", writeAccess, {sym_i, sym_j}, array2dType);

    auto sdfg = builder.move();

    DEBUG_DOT_SDFG(*sdfg);

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& ranges = analysis_manager.get<analysis::MemAccessRanges>();


    // Check result
    auto* range_arg_init = ranges.get("arg_init");
    EXPECT_EQ(range_arg_init, nullptr);

    auto* range_a = ranges.get("A");
    EXPECT_NE(range_a, nullptr);
    EXPECT_EQ(range_a->get_name(), "A");
    EXPECT_FALSE(range_a->saw_read());
    EXPECT_TRUE(range_a->saw_write());
    EXPECT_FALSE(range_a->is_undefined());
    auto& dims = range_a->dims();
    EXPECT_EQ(dims.size(), 2);
    EXPECT_TRUE(symbolic::eq(dims[0].first, symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(dims[0].second, symbolic::integer(22)));
    EXPECT_TRUE(symbolic::eq(dims[1].first, symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(dims[1].second, symbolic::integer(15)));
}

TEST(MemAccessRangeAnalysisTest, Incomplete_2D_Line_Sum) {
    builder::StructuredSDFGBuilder builder("sdfg_incomplete_2d", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Pointer base_ptr_desc(base_desc);

    types::Array array1dType(base_desc, symbolic::symbol("M"));
    types::Pointer array2dType(array1dType);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("B", opaque_desc, true);
    builder.add_container("result", opaque_desc, true);

    builder.add_container("init_i", base_desc);
    builder.add_container("i", base_desc);
    builder.add_container("j", base_desc);
    builder.add_container("sum", base_desc);
    auto sym_i = symbolic::symbol("i");
    auto sym_init_i = symbolic::symbol("init_i");
    auto sym_j = symbolic::symbol("j");

    auto& root = builder.subject().root();
    auto& init_block = builder.add_block(root);
    auto& zero_node = builder.add_constant(init_block, "0", base_desc);
    auto& initTasklet = builder.add_tasklet(init_block, data_flow::TaskletCode::assign, "_out", {"_in"});
    auto& sumInitAccess = builder.add_access(init_block, "sum");
    builder.add_computational_memlet(init_block, zero_node, initTasklet, "_in", {});
    builder.add_computational_memlet(init_block, initTasklet, "_out", sumInitAccess, {});
    auto& b_access = builder.add_access(init_block, "B");
    auto& init_i_tasklet = builder.add_tasklet(init_block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(init_block, b_access, init_i_tasklet, "_in", {symbolic::integer(0)}, base_ptr_desc);
    auto& init_i_access = builder.add_access(init_block, "init_i");
    builder.add_computational_memlet(init_block, init_i_tasklet, "_out", init_i_access, {});


    auto& outer_for = builder.add_for(
        root,
        sym_i,
        symbolic::Eq(symbolic::__false__(), symbolic::Eq(sym_i, symbolic::integer(23))),
        sym_init_i,
        symbolic::add(symbolic::one(), sym_i)
    );
    auto& inner_for = builder.add_for(
        outer_for.root(),
        sym_j,
        symbolic::Lt(sym_j, symbolic::integer(16)),
        symbolic::zero(),
        symbolic::add(symbolic::one(), sym_j)
    );

    auto& inner_block = builder.add_block(inner_for.root());
    auto& tasklet = builder.add_tasklet(inner_block, data_flow::TaskletCode::int_add, "_out", {"_in0", "_in1"});
    auto& prevSumAccess = builder.add_access(inner_block, "sum");
    auto& readPrevSum = builder.add_computational_memlet(inner_block, prevSumAccess, tasklet, "_in0", {});
    auto& readAAccess = builder.add_access(inner_block, "A");
    auto& readArray =
        builder.add_computational_memlet(inner_block, readAAccess, tasklet, "_in1", {sym_i, sym_j}, array2dType);
    auto& writeAccess = builder.add_access(inner_block, "sum");
    builder.add_computational_memlet(inner_block, tasklet, "_out", writeAccess, {});

    auto& result_block = builder.add_block(root);
    auto& sumAccess = builder.add_access(result_block, "sum");
    auto& result_tasklet = builder.add_tasklet(result_block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(result_block, sumAccess, result_tasklet, "_in", {});
    auto& resultAccess = builder.add_access(result_block, "result");
    builder.add_computational_memlet(
        result_block, result_tasklet, "_out", resultAccess, {symbolic::integer(0)}, base_ptr_desc
    );

    auto sdfg = builder.move();

    DEBUG_DOT_SDFG(*sdfg);

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& ranges = analysis_manager.get<analysis::MemAccessRanges>();


    // Check result
    auto* range_arg_init = ranges.get("arg_init");
    EXPECT_EQ(range_arg_init, nullptr);

    auto* range_sum = ranges.get("sum");
    EXPECT_EQ(range_sum, nullptr);

    // Write-pointer to scalar!
    auto* range_result = ranges.get("result");
    EXPECT_NE(range_result, nullptr);
    EXPECT_EQ(range_result->get_name(), "result");
    EXPECT_FALSE(range_result->saw_read());
    EXPECT_TRUE(range_result->saw_write());
    EXPECT_FALSE(range_result->is_undefined());
    auto& dims_res = range_result->dims();
    EXPECT_EQ(dims_res.size(), 1);
    EXPECT_TRUE(symbolic::eq(dims_res[0].first, symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(dims_res[0].second, symbolic::zero()));

    auto* range_a = ranges.get("A");
    EXPECT_NE(range_a, nullptr);
    EXPECT_EQ(range_a->get_name(), "A");
    EXPECT_TRUE(range_a->saw_read());
    EXPECT_FALSE(range_a->saw_write());
    EXPECT_TRUE(range_a->is_undefined());
    auto& dims_a = range_a->dims();
    EXPECT_EQ(dims_a.size(), 2);
    EXPECT_TRUE(dims_a[0].first.is_null());
    EXPECT_TRUE(dims_a[0].second.is_null());

    EXPECT_TRUE(symbolic::eq(dims_a[1].first, symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(dims_a[1].second, symbolic::integer(15)));
}
