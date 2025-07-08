
#include "sdfg/analysis/work_depth_analysis.h"

#include <gtest/gtest.h>
#include <string>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/scalar.h"

using namespace sdfg;

TEST(WorkDepthTest, BlockSingleTasklet) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto desc_type = types::Scalar(types::PrimitiveType::Int32);
    builder.add_container("A", desc_type);

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& tasklet = builder.add_tasklet(
        block,
        data_flow::TaskletCode::assign,
        {"_out", types::Scalar(types::PrimitiveType::Int32)},
        {{"0", types::PrimitiveType::Int32}}
    );
    auto& A = builder.add_access(block, "A");
    builder.add_memlet(block, tasklet, "_out", A, "void", {});

    auto sdfg = builder.move();

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& work_depth_analysis = analysis_manager.get<analysis::WorkDepthAnalysis>();

    auto& root_node = builder_opt.subject().root();
    auto work = work_depth_analysis.work(&root_node);
    auto depth = work_depth_analysis.depth(&root_node);

    EXPECT_TRUE(symbolic::eq(work, symbolic::one()));
    EXPECT_TRUE(symbolic::eq(depth, symbolic::one()));
}

TEST(WorkDepthTest, BlockMultipleTasklets) {
    builder::StructuredSDFGBuilder builder("sdfg_2", FunctionType_CPU);

    auto desc_type = types::Scalar(types::PrimitiveType::Int32);
    builder.add_container("A", desc_type);
    builder.add_container("B", desc_type);

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& tasklet1 = builder.add_tasklet(
        block,
        data_flow::TaskletCode::assign,
        {"_out", types::Scalar(types::PrimitiveType::Int32)},
        {{"0", types::PrimitiveType::Int32}}
    );
    auto& tasklet2 = builder.add_tasklet(
        block,
        data_flow::TaskletCode::add,
        {"_out", types::Scalar(types::PrimitiveType::Int32)},
        {{"_in", types::PrimitiveType::Int32}, {"1", types::PrimitiveType::Int32}}
    );
    auto& A = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "B");
    builder.add_memlet(block, tasklet1, "_out", A, "void", {});
    builder.add_memlet(block, A, "void", tasklet2, "_in", {});
    builder.add_memlet(block, tasklet2, "_out", B, "void", {});

    auto sdfg = builder.move();

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& work_depth_analysis = analysis_manager.get<analysis::WorkDepthAnalysis>();

    auto& root_node = builder_opt.subject().root();
    auto work = work_depth_analysis.work(&root_node);
    auto depth = work_depth_analysis.depth(&root_node);

    EXPECT_TRUE(symbolic::eq(work, symbolic::integer(2)));
    EXPECT_TRUE(symbolic::eq(depth, symbolic::integer(2)));
}

TEST(WorkDepthAnalysis, SequenceMultipleBlocks) {
    builder::StructuredSDFGBuilder builder("sdfg_3", FunctionType_CPU);

    auto desc_type = types::Scalar(types::PrimitiveType::Int32);
    builder.add_container("A", desc_type);
    builder.add_container("B", desc_type);

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root);
    auto& tasklet1 = builder.add_tasklet(
        block1,
        data_flow::TaskletCode::assign,
        {"_out", types::Scalar(types::PrimitiveType::Int32)},
        {{"0", types::PrimitiveType::Int32}}
    );
    auto& A = builder.add_access(block1, "A");
    builder.add_memlet(block1, tasklet1, "_out", A, "void", {});

    auto& block2 = builder.add_block(root);
    auto& tasklet2 = builder.add_tasklet(
        block2,
        data_flow::TaskletCode::add,
        {"_out", types::Scalar(types::PrimitiveType::Int32)},
        {{"_in", types::PrimitiveType::Int32}, {"1", types::PrimitiveType::Int32}}
    );
    auto& A_in = builder.add_access(block2, "A");
    auto& B = builder.add_access(block2, "B");
    builder.add_memlet(block2, A_in, "void", tasklet2, "_in", {});
    builder.add_memlet(block2, tasklet2, "_out", B, "void", {});

    auto sdfg = builder.move();

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& work_depth_analysis = analysis_manager.get<analysis::WorkDepthAnalysis>();

    auto& root_node = builder_opt.subject().root();
    auto work = work_depth_analysis.work(&root_node);
    auto depth = work_depth_analysis.depth(&root_node);

    EXPECT_TRUE(symbolic::eq(work, symbolic::integer(2)));
    EXPECT_TRUE(symbolic::eq(depth, symbolic::integer(2)));
}

TEST(WorkDepthAnalysis, ForLoop) {
    builder::StructuredSDFGBuilder builder("sdfg_4", FunctionType_CPU);

    auto desc_type = types::Scalar(types::PrimitiveType::Int32);
    builder.add_container("A", desc_type);

    auto& root = builder.subject().root();
    auto& for_loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::zero(),
        symbolic::add(symbolic::symbol("i"), symbolic::one())
    );
    auto& block = builder.add_block(for_loop.root());
    auto& tasklet = builder.add_tasklet(
        block,
        data_flow::TaskletCode::assign,
        {"_out", types::Scalar(types::PrimitiveType::Int32)},
        {{"0", types::PrimitiveType::Int32}}
    );
    auto& A = builder.add_access(block, "A");
    builder.add_memlet(block, tasklet, "_out", A, "void", {});

    auto sdfg = builder.move();

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& work_depth_analysis = analysis_manager.get<analysis::WorkDepthAnalysis>();

    auto& root_node = builder_opt.subject().root();
    auto work = work_depth_analysis.work(&root_node);
    auto depth = work_depth_analysis.depth(&root_node);

    EXPECT_TRUE(symbolic::eq(work, symbolic::integer(10)));
    EXPECT_TRUE(symbolic::eq(depth, symbolic::integer(10)));
}

TEST(WorkDepthAnalysis, Map) {
    builder::StructuredSDFGBuilder builder("sdfg_5", FunctionType_CPU);

    auto desc_type = types::Scalar(types::PrimitiveType::Int32);
    builder.add_container("A", desc_type);

    auto& root = builder.subject().root();
    auto& map_node = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::zero(),
        symbolic::add(symbolic::symbol("i"), symbolic::one()),
        ScheduleType_Sequential
    );
    auto& block = builder.add_block(map_node.root());
    auto& tasklet = builder.add_tasklet(
        block,
        data_flow::TaskletCode::assign,
        {"_out", types::Scalar(types::PrimitiveType::Int32)},
        {{"0", types::PrimitiveType::Int32}}
    );
    auto& A = builder.add_access(block, "A");
    builder.add_memlet(block, tasklet, "_out", A, "void", {});

    auto sdfg = builder.move();

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& work_depth_analysis = analysis_manager.get<analysis::WorkDepthAnalysis>();

    auto& root_node = builder_opt.subject().root();
    auto work = work_depth_analysis.work(&root_node);
    auto depth = work_depth_analysis.depth(&root_node);

    EXPECT_TRUE(symbolic::eq(work, symbolic::integer(10)));
    EXPECT_TRUE(symbolic::eq(depth, symbolic::one()));
}

TEST(WorkDepthAnalysis, WhileLoop) {
    builder::StructuredSDFGBuilder builder("sdfg_6", FunctionType_CPU);

    auto desc_type = types::Scalar(types::PrimitiveType::Int32);

    builder.add_container("A", desc_type);
    builder.add_container("B", desc_type);

    auto& root = builder.subject().root();
    auto& while_loop = builder.add_while(root);
    auto& block = builder.add_block(while_loop.root());
    auto& tasklet1 = builder.add_tasklet(
        block,
        data_flow::TaskletCode::assign,
        {"_out", types::Scalar(types::PrimitiveType::Int32)},
        {{"0", types::PrimitiveType::Int32}}
    );
    auto& tasklet2 = builder.add_tasklet(
        block,
        data_flow::TaskletCode::add,
        {"_out", types::Scalar(types::PrimitiveType::Int32)},
        {{"_in", types::PrimitiveType::Int32}, {"1", types::PrimitiveType::Int32}}
    );
    auto& A = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "B");
    builder.add_memlet(block, tasklet1, "_out", A, "void", {});
    builder.add_memlet(block, A, "void", tasklet2, "_in", {});
    builder.add_memlet(block, tasklet2, "_out", B, "void", {});

    auto sdfg = builder.move();

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& work_depth_analysis = analysis_manager.get<analysis::WorkDepthAnalysis>();

    auto& root_node = builder_opt.subject().root();
    auto work = work_depth_analysis.work(&root_node);
    auto depth = work_depth_analysis.depth(&root_node);

    std::string while_symbol_name = "while_" + std::to_string(while_loop.element_id());
    symbolic::Symbol while_symbol = symbolic::symbol(while_symbol_name);

    EXPECT_TRUE(symbolic::eq(work, symbolic::mul(while_symbol, symbolic::integer(2))));
    EXPECT_TRUE(symbolic::eq(depth, symbolic::mul(while_symbol, symbolic::integer(2))));
    EXPECT_TRUE(work_depth_analysis.while_symbols(work).contains(while_symbol));
    EXPECT_EQ(work_depth_analysis.while_symbols(work).size(), 1);
}

TEST(WorkDepthAnalysis, IfElse) {
    builder::StructuredSDFGBuilder builder("sdfg_7", FunctionType_CPU);

    auto desc_type = types::Scalar(types::PrimitiveType::Int32);
    builder.add_container("A", desc_type);
    builder.add_container("B", desc_type);

    auto& root = builder.subject().root();
    auto& if_else = builder.add_if_else(root);
    auto& case_1 = builder.add_case(if_else, symbolic::Lt(symbolic::one(), symbolic::integer(2)));
    auto& block1 = builder.add_block(case_1);
    auto& tasklet1 = builder.add_tasklet(
        block1,
        data_flow::TaskletCode::assign,
        {"_out", types::Scalar(types::PrimitiveType::Int32)},
        {{"0", types::PrimitiveType::Int32}}
    );
    auto& tasklet1_2 = builder.add_tasklet(
        block1,
        data_flow::TaskletCode::add,
        {"_out", types::Scalar(types::PrimitiveType::Int32)},
        {{"_in", types::PrimitiveType::Int32}, {"1", types::PrimitiveType::Int32}}
    );
    auto& A = builder.add_access(block1, "A");
    auto& A_out = builder.add_access(block1, "A");

    builder.add_memlet(block1, tasklet1, "_out", A, "void", {});
    builder.add_memlet(block1, A, "void", tasklet1_2, "_in", {});
    builder.add_memlet(block1, tasklet1_2, "_out", A_out, "void", {});

    auto& case_2 = builder.add_case(if_else, symbolic::Lt(symbolic::one(), symbolic::integer(3)));
    auto& block2 = builder.add_block(case_2);
    auto& tasklet2 = builder.add_tasklet(
        block2,
        data_flow::TaskletCode::add,
        {"_out", types::Scalar(types::PrimitiveType::Int32)},
        {{"_in", types::PrimitiveType::Int32}, {"1", types::PrimitiveType::Int32}}
    );
    auto& A_in_2 = builder.add_access(block2, "A");
    auto& B = builder.add_access(block2, "B");
    builder.add_memlet(block2, A_in_2, "void", tasklet2, "_in", {});
    builder.add_memlet(block2, tasklet2, "_out", B, "void", {});

    auto sdfg = builder.move();

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& work_depth_analysis = analysis_manager.get<analysis::WorkDepthAnalysis>();

    auto& root_node = builder_opt.subject().root();
    auto work = work_depth_analysis.work(&root_node);
    auto depth = work_depth_analysis.depth(&root_node);

    EXPECT_TRUE(symbolic::eq(work, symbolic::max(symbolic::integer(1), symbolic::integer(2))));
    EXPECT_TRUE(symbolic::eq(depth, symbolic::max(symbolic::integer(1), symbolic::integer(2))));
}
