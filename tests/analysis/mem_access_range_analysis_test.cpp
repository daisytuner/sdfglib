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

TEST(MemAccessRangeTest, Arg_Index_Write) {
    builder::StructuredSDFGBuilder builder("sdfg_arg_idx", FunctionType_CPU);

    auto dataType = types::Scalar(types::PrimitiveType::Int32);
    auto& array1dType = static_cast<const types::IType&>(types::Pointer(dataType));

    builder.add_container("A", array1dType, true);
    builder.add_container("arg_idx", dataType, true);

    auto symA = symbolic::symbol("A");
    auto sym_idx = symbolic::symbol("arg_idx");

    auto& root = builder.subject().root();

    auto& block = builder.add_block(root);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"__out", dataType}, {{"0", dataType}});
    auto& writeAccess = builder.add_access(block, "A");
    auto& writeArg = builder.add_memlet(block, tasklet, "__out", writeAccess, "void", {sym_idx});

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
    EXPECT_EQ(dims.size(), 1);
    EXPECT_TRUE(symbolic::eq(dims[0].first, symbolic::symbol("arg_idx")));
    EXPECT_TRUE(symbolic::eq(dims[0].second, symbolic::symbol("arg_idx")));
}

TEST(MemAccessRangeTest, Simple_2D_Map_Init) {
    builder::StructuredSDFGBuilder builder("sdfg_simple_2d", FunctionType_CPU);

    auto dataType = types::Scalar(types::PrimitiveType::Int32);
    auto array2dType = types::Pointer(static_cast<const types::IType&>(types::Pointer(dataType)));

    builder.add_container("A", array2dType, true);
    builder.add_container("arg_init", dataType, true);
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("j", types::Scalar(types::PrimitiveType::Int32));
    auto symA = symbolic::symbol("A");
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
    auto& tasklet =
        builder.add_tasklet(block, data_flow::TaskletCode::assign, {"__out", dataType}, {{"__in", dataType}});
    auto& readAccess = builder.add_access(block, "arg_init");
    auto& readArg = builder.add_memlet(block, readAccess, "void", tasklet, "__in", {});
    auto& writeAccess = builder.add_access(block, "A");
    auto& writeArg = builder.add_memlet(block, tasklet, "__out", writeAccess, "void", {sym_i, sym_j});

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

TEST(MemAccessRangeTest, Incomplete_2D_Line_Sum) {
    builder::StructuredSDFGBuilder builder("sdfg_incomplete_2d", FunctionType_CPU);

    auto dataType = types::Scalar(types::PrimitiveType::Int32);
    auto array2dType = types::Pointer(static_cast<const types::IType&>(types::Pointer(dataType)));

    builder.add_container("A", array2dType, true);
    builder.add_container("B", types::Pointer(dataType), true);
    builder.add_container("init_i", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("j", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("sum", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("result", types::Pointer(types::Scalar(types::PrimitiveType::Int32)), true);
    auto symA = symbolic::symbol("A");
    auto sym_i = symbolic::symbol("i");
    auto sym_init_i = symbolic::symbol("init_i");
    auto sym_j = symbolic::symbol("j");

    auto& root = builder.subject().root();
    auto& init_block = builder.add_block(root);
    auto& initTasklet =
        builder.add_tasklet(init_block, data_flow::TaskletCode::assign, {"__out", dataType}, {{"0", dataType}});
    auto& sumInitAccess = builder.add_access(init_block, "sum");
    builder.add_memlet(init_block, initTasklet, "__out", sumInitAccess, "void", {});
    auto& b_access = builder.add_access(init_block, "B");
    auto& init_i_tasklet =
        builder.add_tasklet(init_block, data_flow::TaskletCode::assign, {"__out", dataType}, {{"__in", dataType}});
    builder.add_memlet(init_block, b_access, "void", init_i_tasklet, "__in", {symbolic::integer(0)});
    auto& init_i_access = builder.add_access(init_block, "init_i");
    builder.add_memlet(init_block, init_i_tasklet, "__out", init_i_access, "void", {});


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
    auto& tasklet = builder.add_tasklet(
        inner_block, data_flow::TaskletCode::add, {"__out", dataType}, {{"__in0", dataType}, {"__in1", dataType}}
    );
    auto& prevSumAccess = builder.add_access(inner_block, "sum");
    auto& readPrevSum = builder.add_memlet(inner_block, prevSumAccess, "void", tasklet, "__in0", {});
    auto& readAAccess = builder.add_access(inner_block, "A");
    auto& readArray = builder.add_memlet(inner_block, readAAccess, "void", tasklet, "__in1", {sym_i, sym_j});
    auto& writeAccess = builder.add_access(inner_block, "sum");
    builder.add_memlet(inner_block, tasklet, "__out", writeAccess, "void", {});

    auto& result_block = builder.add_block(root);
    auto& sumAccess = builder.add_access(result_block, "sum");
    auto& result_tasklet =
        builder.add_tasklet(result_block, data_flow::TaskletCode::assign, {"__out", dataType}, {{"__in", dataType}});
    builder.add_memlet(result_block, sumAccess, "void", result_tasklet, "__in", {});
    auto& resultAccess = builder.add_access(result_block, "result");
    builder.add_memlet(result_block, result_tasklet, "__out", resultAccess, "void", {symbolic::integer(0)});

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
