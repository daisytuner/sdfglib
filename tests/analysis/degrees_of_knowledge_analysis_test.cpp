
#include "sdfg/analysis/degrees_of_knowledge_analysis.h"

#include <gtest/gtest.h>
#include <string>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/array.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"

using namespace sdfg;

TEST(DOKSizeTest, StaticSize) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto desc_type = types::Scalar(types::PrimitiveType::Int32);
    builder.add_container("i", desc_type);

    auto& root = builder.subject().root();

    symbolic::Symbol indvar = symbolic::symbol("i");
    symbolic::Condition condition = symbolic::Lt(indvar, symbolic::integer(10));
    symbolic::Expression init = symbolic::zero();
    symbolic::Expression update = symbolic::add(indvar, symbolic::one());
    ScheduleType schedule_type = ScheduleType_Sequential::create();

    auto& map_node = builder.add_map(root, indvar, condition, init, update, schedule_type);

    auto& sdfg = builder.subject();

    // Run analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& work_depth_analysis = analysis_manager.get<analysis::DegreesOfKnowledgeAnalysis>();

    auto& root_node = sdfg.root();
    auto work = work_depth_analysis.size_of_a_map(map_node);
    EXPECT_EQ(work.second, analysis::DegreesOfKnowledgeClassification::Scalar);
    EXPECT_TRUE(symbolic::eq(work.first, symbolic::integer(10)));
}

TEST(DOKSizeTest, StaticSizeOffset) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto desc_type = types::Scalar(types::PrimitiveType::Int32);
    builder.add_container("i", desc_type);

    auto& root = builder.subject().root();

    symbolic::Symbol indvar = symbolic::symbol("i");
    symbolic::Condition condition = symbolic::Lt(indvar, symbolic::integer(10));
    symbolic::Expression init = symbolic::integer(5);
    symbolic::Expression update = symbolic::add(indvar, symbolic::one());
    ScheduleType schedule_type = ScheduleType_Sequential::create();

    auto& map_node = builder.add_map(root, indvar, condition, init, update, schedule_type);

    auto& sdfg = builder.subject();

    // Run analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& work_depth_analysis = analysis_manager.get<analysis::DegreesOfKnowledgeAnalysis>();

    auto& root_node = sdfg.root();
    auto work = work_depth_analysis.size_of_a_map(map_node);
    EXPECT_EQ(work.second, analysis::DegreesOfKnowledgeClassification::Scalar);
    EXPECT_TRUE(symbolic::eq(work.first, symbolic::integer(5)));
}

TEST(DOKSizeTest, StaticSizeStride) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto desc_type = types::Scalar(types::PrimitiveType::Int32);
    builder.add_container("i", desc_type);

    auto& root = builder.subject().root();

    symbolic::Symbol indvar = symbolic::symbol("i");
    symbolic::Condition condition = symbolic::Lt(indvar, symbolic::integer(10));
    symbolic::Expression init = symbolic::zero();
    symbolic::Expression update = symbolic::add(indvar, symbolic::integer(2));
    ScheduleType schedule_type = ScheduleType_Sequential::create();

    auto& map_node = builder.add_map(root, indvar, condition, init, update, schedule_type);

    auto& sdfg = builder.subject();

    // Run analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& work_depth_analysis = analysis_manager.get<analysis::DegreesOfKnowledgeAnalysis>();

    auto& root_node = sdfg.root();
    auto work = work_depth_analysis.size_of_a_map(map_node);
    EXPECT_EQ(work.second, analysis::DegreesOfKnowledgeClassification::Scalar);
    EXPECT_TRUE(symbolic::eq(work.first, symbolic::integer(5)));
}

TEST(DOKSizeTest, BoundSize) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto desc_type = types::Scalar(types::PrimitiveType::Int32);
    builder.add_container("i", desc_type);
    builder.add_container("N", desc_type);

    auto& root = builder.subject().root();

    symbolic::Symbol indvar = symbolic::symbol("i");
    symbolic::Condition condition = symbolic::Lt(indvar, symbolic::symbol("N"));
    symbolic::Expression init = symbolic::zero();
    symbolic::Expression update = symbolic::add(indvar, symbolic::one());
    ScheduleType schedule_type = ScheduleType_Sequential::create();

    auto& map_node = builder.add_map(root, indvar, condition, init, update, schedule_type);

    auto& sdfg = builder.subject();

    // Run analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& work_depth_analysis = analysis_manager.get<analysis::DegreesOfKnowledgeAnalysis>();

    auto& root_node = sdfg.root();
    auto work = work_depth_analysis.size_of_a_map(map_node);
    EXPECT_EQ(work.second, analysis::DegreesOfKnowledgeClassification::Bound);
    EXPECT_TRUE(symbolic::eq(work.first, symbolic::symbol("N")));
}

TEST(DOKSizeTest, UnboundSize) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto desc_type = types::Scalar(types::PrimitiveType::Int32);
    builder.add_container("i", desc_type);
    builder.add_container("j", desc_type);
    builder.add_container("N", desc_type);

    auto& root = builder.subject().root();

    symbolic::Symbol indvar_outer = symbolic::symbol("j");
    symbolic::Condition condition_outer = symbolic::Lt(indvar_outer, symbolic::symbol("N"));
    symbolic::Expression init_outer = symbolic::zero();
    symbolic::Expression update_outer = symbolic::add(indvar_outer, symbolic::one());
    ScheduleType schedule_type_outer = ScheduleType_Sequential::create();

    auto& map_node_outer =
        builder.add_map(root, indvar_outer, condition_outer, init_outer, update_outer, schedule_type_outer);

    symbolic::Symbol indvar = symbolic::symbol("i");
    symbolic::Condition condition = symbolic::Lt(indvar, symbolic::symbol("j"));
    symbolic::Expression init = symbolic::zero();
    symbolic::Expression update = symbolic::add(indvar, symbolic::one());
    ScheduleType schedule_type = ScheduleType_Sequential::create();

    auto& map_node = builder.add_map(map_node_outer.root(), indvar, condition, init, update, schedule_type);

    auto& sdfg = builder.subject();

    // Run analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& work_depth_analysis = analysis_manager.get<analysis::DegreesOfKnowledgeAnalysis>();

    auto& root_node = sdfg.root();
    auto work = work_depth_analysis.size_of_a_map(map_node);
    EXPECT_EQ(work.second, analysis::DegreesOfKnowledgeClassification::Unbound);
    EXPECT_TRUE(symbolic::eq(work.first, symbolic::symbol("j")));
}

TEST(DOKNumberTest, StaticNumber) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto desc_type = types::Scalar(types::PrimitiveType::Int32);
    builder.add_container("i", desc_type);
    builder.add_container("j", desc_type);

    auto& root = builder.subject().root();

    symbolic::Symbol indvar_outer = symbolic::symbol("j");
    symbolic::Condition condition_outer = symbolic::Lt(indvar_outer, symbolic::integer(10));
    symbolic::Expression init_outer = symbolic::zero();
    symbolic::Expression update_outer = symbolic::add(indvar_outer, symbolic::one());
    ScheduleType schedule_type_outer = ScheduleType_Sequential::create();

    auto& map_node_outer =
        builder.add_map(root, indvar_outer, condition_outer, init_outer, update_outer, schedule_type_outer);

    symbolic::Symbol indvar = symbolic::symbol("i");
    symbolic::Condition condition = symbolic::Lt(indvar, symbolic::integer(10));
    symbolic::Expression init = symbolic::zero();
    symbolic::Expression update = symbolic::add(indvar, symbolic::one());
    ScheduleType schedule_type = ScheduleType_Sequential::create();

    auto& map_node = builder.add_map(map_node_outer.root(), indvar, condition, init, update, schedule_type);

    auto& sdfg = builder.subject();

    // Run analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& work_depth_analysis = analysis_manager.get<analysis::DegreesOfKnowledgeAnalysis>();

    auto& root_node = sdfg.root();
    auto work = work_depth_analysis.number_of_maps(map_node);
    EXPECT_EQ(work.second, analysis::DegreesOfKnowledgeClassification::Scalar);
    EXPECT_TRUE(symbolic::eq(work.first, symbolic::integer(10)));
}

TEST(DOKNumberTest, StaticNumberOffset) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto desc_type = types::Scalar(types::PrimitiveType::Int32);
    builder.add_container("i", desc_type);
    builder.add_container("j", desc_type);

    auto& root = builder.subject().root();

    symbolic::Symbol indvar_outer = symbolic::symbol("j");
    symbolic::Condition condition_outer = symbolic::Lt(indvar_outer, symbolic::integer(10));
    symbolic::Expression init_outer = symbolic::integer(5);
    symbolic::Expression update_outer = symbolic::add(indvar_outer, symbolic::one());
    ScheduleType schedule_type_outer = ScheduleType_Sequential::create();

    auto& map_node_outer =
        builder.add_map(root, indvar_outer, condition_outer, init_outer, update_outer, schedule_type_outer);

    symbolic::Symbol indvar = symbolic::symbol("i");
    symbolic::Condition condition = symbolic::Lt(indvar, symbolic::integer(10));
    symbolic::Expression init = symbolic::zero();
    symbolic::Expression update = symbolic::add(indvar, symbolic::one());
    ScheduleType schedule_type = ScheduleType_Sequential::create();

    auto& map_node = builder.add_map(map_node_outer.root(), indvar, condition, init, update, schedule_type);

    auto& sdfg = builder.subject();

    // Run analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& work_depth_analysis = analysis_manager.get<analysis::DegreesOfKnowledgeAnalysis>();

    auto& root_node = sdfg.root();
    auto work = work_depth_analysis.number_of_maps(map_node);
    EXPECT_EQ(work.second, analysis::DegreesOfKnowledgeClassification::Scalar);
    EXPECT_TRUE(symbolic::eq(work.first, symbolic::integer(5)));
}

TEST(DOKNumberTest, StaticNumberStride) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto desc_type = types::Scalar(types::PrimitiveType::Int32);
    builder.add_container("i", desc_type);
    builder.add_container("j", desc_type);

    auto& root = builder.subject().root();

    symbolic::Symbol indvar_outer = symbolic::symbol("j");
    symbolic::Condition condition_outer = symbolic::Lt(indvar_outer, symbolic::integer(10));
    symbolic::Expression init_outer = symbolic::zero();
    symbolic::Expression update_outer = symbolic::add(indvar_outer, symbolic::integer(2));
    ScheduleType schedule_type_outer = ScheduleType_Sequential::create();

    auto& map_node_outer =
        builder.add_map(root, indvar_outer, condition_outer, init_outer, update_outer, schedule_type_outer);

    symbolic::Symbol indvar = symbolic::symbol("i");
    symbolic::Condition condition = symbolic::Lt(indvar, symbolic::integer(10));
    symbolic::Expression init = symbolic::zero();
    symbolic::Expression update = symbolic::add(indvar, symbolic::one());
    ScheduleType schedule_type = ScheduleType_Sequential::create();

    auto& map_node = builder.add_map(map_node_outer.root(), indvar, condition, init, update, schedule_type);

    auto& sdfg = builder.subject();

    // Run analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& work_depth_analysis = analysis_manager.get<analysis::DegreesOfKnowledgeAnalysis>();

    auto& root_node = sdfg.root();
    auto work = work_depth_analysis.number_of_maps(map_node);
    EXPECT_EQ(work.second, analysis::DegreesOfKnowledgeClassification::Scalar);
    EXPECT_TRUE(symbolic::eq(work.first, symbolic::integer(5)));
}

TEST(DOKNumberTest, BoundNumber) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto desc_type = types::Scalar(types::PrimitiveType::Int32);
    builder.add_container("i", desc_type);
    builder.add_container("j", desc_type);
    builder.add_container("N", desc_type);

    auto& root = builder.subject().root();

    symbolic::Symbol indvar_outer = symbolic::symbol("j");
    symbolic::Condition condition_outer = symbolic::Lt(indvar_outer, symbolic::symbol("N"));
    symbolic::Expression init_outer = symbolic::zero();
    symbolic::Expression update_outer = symbolic::add(indvar_outer, symbolic::one());
    ScheduleType schedule_type_outer = ScheduleType_Sequential::create();

    auto& map_node_outer =
        builder.add_map(root, indvar_outer, condition_outer, init_outer, update_outer, schedule_type_outer);

    symbolic::Symbol indvar = symbolic::symbol("i");
    symbolic::Condition condition = symbolic::Lt(indvar, symbolic::integer(10));
    symbolic::Expression init = symbolic::zero();
    symbolic::Expression update = symbolic::add(indvar, symbolic::one());
    ScheduleType schedule_type = ScheduleType_Sequential::create();

    auto& map_node = builder.add_map(map_node_outer.root(), indvar, condition, init, update, schedule_type);

    auto& sdfg = builder.subject();

    // Run analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& work_depth_analysis = analysis_manager.get<analysis::DegreesOfKnowledgeAnalysis>();

    auto& root_node = sdfg.root();
    auto work = work_depth_analysis.number_of_maps(map_node);
    EXPECT_EQ(work.second, analysis::DegreesOfKnowledgeClassification::Bound);
    EXPECT_TRUE(symbolic::eq(work.first, symbolic::symbol("N")));
}

TEST(DOKNumberTest, UnboundNumber) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto desc_type = types::Scalar(types::PrimitiveType::Int32);
    builder.add_container("i", desc_type);
    builder.add_container("j", desc_type);

    auto& root = builder.subject().root();

    auto& while_node = builder.add_while(root);

    symbolic::Symbol indvar = symbolic::symbol("i");
    symbolic::Condition condition = symbolic::Lt(indvar, symbolic::integer(10));
    symbolic::Expression init = symbolic::zero();
    symbolic::Expression update = symbolic::add(indvar, symbolic::one());
    ScheduleType schedule_type = ScheduleType_Sequential::create();

    auto& map_node = builder.add_map(while_node.root(), indvar, condition, init, update, schedule_type);

    std::string while_symbol_name = "while_" + std::to_string(while_node.element_id());

    auto& sdfg = builder.subject();

    // Run analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& work_depth_analysis = analysis_manager.get<analysis::DegreesOfKnowledgeAnalysis>();

    auto& root_node = sdfg.root();
    auto work = work_depth_analysis.number_of_maps(map_node);
    EXPECT_EQ(work.second, analysis::DegreesOfKnowledgeClassification::Unbound);
    EXPECT_TRUE(symbolic::eq(work.first, symbolic::symbol(while_symbol_name)));
}

TEST(DOKLoadTest, StaticLoad) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto desc_type = types::Scalar(types::PrimitiveType::Int32);
    auto sc_type = types::Scalar(types::PrimitiveType::Float);
    auto pt_type = types::Pointer(sc_type);
    builder.add_container("i", desc_type);
    builder.add_container("A", pt_type);
    builder.add_container("B", pt_type);

    auto& root = builder.subject().root();

    symbolic::Symbol indvar = symbolic::symbol("i");
    symbolic::Condition condition = symbolic::Lt(indvar, symbolic::integer(10));
    symbolic::Expression init = symbolic::zero();
    symbolic::Expression update = symbolic::add(indvar, symbolic::one());
    ScheduleType schedule_type = ScheduleType_Sequential::create();

    auto& map_node = builder.add_map(root, indvar, condition, init, update, schedule_type);

    auto& block = builder.add_block(map_node.root());
    auto& access_A = builder.add_access(block, "A");
    auto& access_B = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});

    builder.add_computational_memlet(block, access_A, tasklet, "_in", {symbolic::symbol("i")});
    builder.add_computational_memlet(block, tasklet, "_out", access_B, {symbolic::symbol("i")});

    auto& sdfg = builder.subject();

    // Run analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& work_depth_analysis = analysis_manager.get<analysis::DegreesOfKnowledgeAnalysis>();

    auto& root_node = sdfg.root();
    auto work = work_depth_analysis.load_of_a_map(map_node);
    EXPECT_EQ(work.second, analysis::DegreesOfKnowledgeClassification::Scalar);
    EXPECT_TRUE(symbolic::eq(work.first, symbolic::integer(1)));
}

TEST(DOKLoadTest, BoundLoad) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto desc_type = types::Scalar(types::PrimitiveType::Int32);
    auto sc_type = types::Scalar(types::PrimitiveType::Float);
    auto pt_type = types::Array(sc_type, symbolic::symbol("N"));
    auto pt2_type = types::Pointer(pt_type);
    builder.add_container("i", desc_type);
    builder.add_container("N", desc_type);
    builder.add_container("j", desc_type);
    builder.add_container("A", pt2_type);
    builder.add_container("B", pt2_type);

    auto& root = builder.subject().root();

    symbolic::Symbol indvar = symbolic::symbol("i");
    symbolic::Condition condition = symbolic::Lt(indvar, symbolic::integer(10));
    symbolic::Expression init = symbolic::zero();
    symbolic::Expression update = symbolic::add(indvar, symbolic::one());
    ScheduleType schedule_type = ScheduleType_Sequential::create();

    auto& map_node = builder.add_map(root, indvar, condition, init, update, schedule_type);

    symbolic::Symbol indvar_inner = symbolic::symbol("j");
    symbolic::Condition condition_inner = symbolic::Lt(indvar_inner, symbolic::symbol("N"));
    symbolic::Expression init_inner = symbolic::zero();
    symbolic::Expression update_inner = symbolic::add(indvar_inner, symbolic::one());
    ScheduleType schedule_type_inner = ScheduleType_Sequential::create();

    auto& map_node_inner =
        builder.add_map(map_node.root(), indvar_inner, condition_inner, init_inner, update_inner, schedule_type_inner);

    auto& block = builder.add_block(map_node_inner.root());
    auto& access_A = builder.add_access(block, "A");
    auto& access_B = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});

    builder.add_computational_memlet(block, access_A, tasklet, "_in", {symbolic::symbol("i"), symbolic::symbol("j")});
    builder.add_computational_memlet(block, tasklet, "_out", access_B, {symbolic::symbol("i"), symbolic::symbol("j")});

    auto& sdfg = builder.subject();

    // Run analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& work_depth_analysis = analysis_manager.get<analysis::DegreesOfKnowledgeAnalysis>();

    auto& root_node = sdfg.root();
    auto work = work_depth_analysis.load_of_a_map(map_node);
    EXPECT_EQ(work.second, analysis::DegreesOfKnowledgeClassification::Bound);
    EXPECT_TRUE(symbolic::eq(work.first, symbolic::symbol("N")));
}

TEST(DOKLoadTest, UnboundLoad) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto desc_type = types::Scalar(types::PrimitiveType::Int32);
    auto sc_type = types::Scalar(types::PrimitiveType::Float);
    auto pt_type = types::Array(sc_type, symbolic::symbol("N"));
    auto pt2_type = types::Pointer(pt_type);
    builder.add_container("i", desc_type);
    builder.add_container("N", desc_type);
    builder.add_container("j", desc_type);
    builder.add_container("A", pt2_type);
    builder.add_container("B", pt2_type);

    auto& root = builder.subject().root();

    symbolic::Symbol indvar = symbolic::symbol("i");
    symbolic::Condition condition = symbolic::Lt(indvar, symbolic::integer(10));
    symbolic::Expression init = symbolic::zero();
    symbolic::Expression update = symbolic::add(indvar, symbolic::one());
    ScheduleType schedule_type = ScheduleType_Sequential::create();

    auto& map_node = builder.add_map(root, indvar, condition, init, update, schedule_type);

    auto& block_init = builder.add_block(map_node.root());
    auto& tasklet_init = builder.add_tasklet(block_init, data_flow::TaskletCode::assign, "_out", {"0"});
    auto& access_j = builder.add_access(block_init, "j");
    builder.add_computational_memlet(block_init, tasklet_init, "_out", access_j, {});

    auto& while_node = builder.add_while(map_node.root());

    auto& if_else_node = builder.add_if_else(while_node.root());
    auto& loop_end = builder.add_case(if_else_node, symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("N")));
    builder.add_break(loop_end);

    auto& block = builder.add_block(while_node.root());
    auto& access_A = builder.add_access(block, "A");
    auto& access_B = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});

    builder.add_computational_memlet(block, access_A, tasklet, "_in", {symbolic::symbol("i"), symbolic::symbol("j")});
    builder.add_computational_memlet(block, tasklet, "_out", access_B, {symbolic::symbol("i"), symbolic::symbol("j")});

    auto& update_read = builder.add_access(block, "j");
    auto& update_write = builder.add_access(block, "j");
    auto& update_tasklet = builder.add_tasklet(block, data_flow::TaskletCode::add, "_out", {"_in", "1"});

    builder.add_computational_memlet(block, update_read, update_tasklet, "_in", {});
    builder.add_computational_memlet(block, update_tasklet, "_out", update_write, {});

    std::string while_symbol_name = "while_" + std::to_string(while_node.element_id());

    auto& sdfg = builder.subject();

    // Run analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& work_depth_analysis = analysis_manager.get<analysis::DegreesOfKnowledgeAnalysis>();

    auto& root_node = sdfg.root();
    auto work = work_depth_analysis.load_of_a_map(map_node);
    EXPECT_EQ(work.second, analysis::DegreesOfKnowledgeClassification::Unbound);
    EXPECT_TRUE(symbolic::eq(
        work.first,
        symbolic::add(symbolic::one(), symbolic::mul(symbolic::symbol(while_symbol_name), symbolic::integer(2)))
    ));
}

TEST(DOKBalanceTest, StaticBalance) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto desc_type = types::Scalar(types::PrimitiveType::Int32);
    auto sc_type = types::Scalar(types::PrimitiveType::Float);
    auto pt_type = types::Pointer(sc_type);
    builder.add_container("i", desc_type);
    builder.add_container("A", pt_type);
    builder.add_container("B", pt_type);

    auto& root = builder.subject().root();

    symbolic::Symbol indvar = symbolic::symbol("i");
    symbolic::Condition condition = symbolic::Lt(indvar, symbolic::integer(10));
    symbolic::Expression init = symbolic::zero();
    symbolic::Expression update = symbolic::add(indvar, symbolic::one());
    ScheduleType schedule_type = ScheduleType_Sequential::create();

    auto& map_node = builder.add_map(root, indvar, condition, init, update, schedule_type);

    auto& block = builder.add_block(map_node.root());
    auto& access_A = builder.add_access(block, "A");
    auto& access_B = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});

    builder.add_computational_memlet(block, access_A, tasklet, "_in", {symbolic::symbol("i")});
    builder.add_computational_memlet(block, tasklet, "_out", access_B, {symbolic::symbol("i")});

    auto& sdfg = builder.subject();

    // Run analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& work_depth_analysis = analysis_manager.get<analysis::DegreesOfKnowledgeAnalysis>();

    auto& root_node = sdfg.root();
    auto work = work_depth_analysis.balance_of_a_map(map_node);
    EXPECT_EQ(work.second, analysis::DegreesOfKnowledgeClassification::Scalar);
    EXPECT_TRUE(symbolic::eq(work.first, symbolic::integer(2)));
}

TEST(DOKBalanceTest, BoundBalance) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto desc_type = types::Scalar(types::PrimitiveType::Int32);
    auto sc_type = types::Scalar(types::PrimitiveType::Float);
    auto pt_type = types::Array(sc_type, symbolic::integer(10));
    auto pt2_type = types::Pointer(pt_type);
    builder.add_container("i", desc_type);
    builder.add_container("j", desc_type);
    builder.add_container("A", pt2_type);
    builder.add_container("B", pt2_type);

    auto& root = builder.subject().root();

    symbolic::Symbol indvar = symbolic::symbol("i");
    symbolic::Condition condition = symbolic::Lt(indvar, symbolic::integer(10));
    symbolic::Expression init = symbolic::zero();
    symbolic::Expression update = symbolic::add(indvar, symbolic::one());
    ScheduleType schedule_type = ScheduleType_Sequential::create();

    auto& map_node = builder.add_map(root, indvar, condition, init, update, schedule_type);

    symbolic::Symbol indvar_inner = symbolic::symbol("j");
    symbolic::Condition condition_inner = symbolic::Lt(indvar_inner, symbolic::symbol("i"));
    symbolic::Expression init_inner = symbolic::zero();
    symbolic::Expression update_inner = symbolic::add(indvar_inner, symbolic::one());
    ScheduleType schedule_type_inner = ScheduleType_Sequential::create();

    auto& map_node_inner =
        builder.add_map(map_node.root(), indvar_inner, condition_inner, init_inner, update_inner, schedule_type_inner);

    auto& block = builder.add_block(map_node_inner.root());
    auto& access_A = builder.add_access(block, "A");
    auto& access_B = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});

    builder.add_computational_memlet(block, access_A, tasklet, "_in", {symbolic::symbol("i"), symbolic::symbol("j")});
    builder.add_computational_memlet(block, tasklet, "_out", access_B, {symbolic::symbol("i"), symbolic::symbol("j")});

    auto& sdfg = builder.subject();

    // Run analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& work_depth_analysis = analysis_manager.get<analysis::DegreesOfKnowledgeAnalysis>();

    auto& root_node = sdfg.root();
    auto work = work_depth_analysis.balance_of_a_map(map_node);
    EXPECT_EQ(work.second, analysis::DegreesOfKnowledgeClassification::Bound);
    EXPECT_TRUE(symbolic::
                    eq(work.first,
                       symbolic::add(symbolic::mul(symbolic::integer(2), symbolic::symbol("i")), symbolic::one())));
}

TEST(DOKBalanceTest, UnboundBalance) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto desc_type = types::Scalar(types::PrimitiveType::Int32);
    auto sc_type = types::Scalar(types::PrimitiveType::Float);
    auto pt_type = types::Array(sc_type, symbolic::integer(100));
    auto pt2_type = types::Pointer(pt_type);
    builder.add_container("i", desc_type);
    builder.add_container("j", desc_type);
    builder.add_container("A", pt2_type);
    builder.add_container("B", pt2_type);

    auto& root = builder.subject().root();

    symbolic::Symbol indvar = symbolic::symbol("i");
    symbolic::Condition condition = symbolic::Lt(indvar, symbolic::integer(10));
    symbolic::Expression init = symbolic::zero();
    symbolic::Expression update = symbolic::add(indvar, symbolic::one());
    ScheduleType schedule_type = ScheduleType_Sequential::create();

    auto& map_node = builder.add_map(root, indvar, condition, init, update, schedule_type);

    auto& block_init = builder.add_block(map_node.root());
    auto& tasklet_init = builder.add_tasklet(block_init, data_flow::TaskletCode::assign, "_out", {"0"});
    auto& access_j = builder.add_access(block_init, "j");
    builder.add_computational_memlet(block_init, tasklet_init, "_out", access_j, {});

    auto& while_node = builder.add_while(map_node.root());

    auto& if_else_node = builder.add_if_else(while_node.root());
    auto& loop_end = builder.add_case(
        if_else_node, symbolic::Lt(symbolic::mul(symbolic::symbol("i"), symbolic::symbol("j")), symbolic::integer(100))
    );
    builder.add_break(loop_end);

    auto& block = builder.add_block(while_node.root());
    auto& access_A = builder.add_access(block, "A");
    auto& access_B = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});

    builder.add_computational_memlet(block, access_A, tasklet, "_in", {symbolic::symbol("i"), symbolic::symbol("j")});
    builder.add_computational_memlet(block, tasklet, "_out", access_B, {symbolic::symbol("i"), symbolic::symbol("j")});

    auto& update_read = builder.add_access(block, "j");
    auto& update_write = builder.add_access(block, "j");
    auto& update_tasklet = builder.add_tasklet(block, data_flow::TaskletCode::add, "_out", {"_in", "1"});

    builder.add_computational_memlet(block, update_read, update_tasklet, "_in", {});
    builder.add_computational_memlet(block, update_tasklet, "_out", update_write, {});

    std::string while_symbol_name = "while_" + std::to_string(while_node.element_id());
    std::string if_else_symbol_name = "if_else_" + std::to_string(if_else_node.element_id());

    auto& sdfg = builder.subject();

    // Run analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& work_depth_analysis = analysis_manager.get<analysis::DegreesOfKnowledgeAnalysis>();

    auto& root_node = sdfg.root();
    auto work = work_depth_analysis.balance_of_a_map(map_node);
    auto expected_balance = symbolic::add(
        symbolic::integer(2),
        symbolic::mul(
            symbolic::symbol(while_symbol_name),
            symbolic::add(symbolic::integer(2), symbolic::mul(symbolic::integer(2), symbolic::symbol(if_else_symbol_name)))
        )
    );

    EXPECT_EQ(work.second, analysis::DegreesOfKnowledgeClassification::Unbound);
    EXPECT_TRUE(symbolic::eq(work.first, expected_balance));
}
