
#include "sdfg/analysis/degrees_of_knowledge_analysis.h"

#include <gtest/gtest.h>
#include <string>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/symbolic/symbolic.h"
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
    ScheduleType schedule_type = ScheduleType_Sequential;

    auto& map_node = builder.add_map(root, indvar, condition, init, update, schedule_type);

    auto sdfg = builder.move();

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& work_depth_analysis = analysis_manager.get<analysis::DegreesOfKnowledgeAnalysis>();

    auto& root_node = builder_opt.subject().root();
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
    ScheduleType schedule_type = ScheduleType_Sequential;

    auto& map_node = builder.add_map(root, indvar, condition, init, update, schedule_type);

    auto sdfg = builder.move();

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& work_depth_analysis = analysis_manager.get<analysis::DegreesOfKnowledgeAnalysis>();

    auto& root_node = builder_opt.subject().root();
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
    ScheduleType schedule_type = ScheduleType_Sequential;

    auto& map_node = builder.add_map(root, indvar, condition, init, update, schedule_type);

    auto sdfg = builder.move();

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& work_depth_analysis = analysis_manager.get<analysis::DegreesOfKnowledgeAnalysis>();

    auto& root_node = builder_opt.subject().root();
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
    ScheduleType schedule_type = ScheduleType_Sequential;

    auto& map_node = builder.add_map(root, indvar, condition, init, update, schedule_type);

    auto sdfg = builder.move();

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& work_depth_analysis = analysis_manager.get<analysis::DegreesOfKnowledgeAnalysis>();

    auto& root_node = builder_opt.subject().root();
    auto work = work_depth_analysis.size_of_a_map(map_node);
    EXPECT_EQ(work.second, analysis::DegreesOfKnowledgeClassification::Bound);
    EXPECT_TRUE(symbolic::eq(work.first, symbolic::symbol("N")));
}

TEST(DOKSizeTest, UnboundSize) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto desc_type = types::Scalar(types::PrimitiveType::Int32);
    builder.add_container("i", desc_type);
    builder.add_container("j", desc_type);

    auto& root = builder.subject().root();

    symbolic::Symbol indvar_outer = symbolic::symbol("j");
    symbolic::Condition condition_outer = symbolic::Lt(indvar_outer, symbolic::symbol("N"));
    symbolic::Expression init_outer = symbolic::zero();
    symbolic::Expression update_outer = symbolic::add(indvar_outer, symbolic::one());
    ScheduleType schedule_type_outer = ScheduleType_Sequential;

    auto& map_node_outer =
        builder.add_map(root, indvar_outer, condition_outer, init_outer, update_outer, schedule_type_outer);

    symbolic::Symbol indvar = symbolic::symbol("i");
    symbolic::Condition condition = symbolic::Lt(indvar, symbolic::symbol("j"));
    symbolic::Expression init = symbolic::zero();
    symbolic::Expression update = symbolic::add(indvar, symbolic::one());
    ScheduleType schedule_type = ScheduleType_Sequential;

    auto& map_node = builder.add_map(map_node_outer.root(), indvar, condition, init, update, schedule_type);

    auto sdfg = builder.move();

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& work_depth_analysis = analysis_manager.get<analysis::DegreesOfKnowledgeAnalysis>();

    auto& root_node = builder_opt.subject().root();
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
    ScheduleType schedule_type_outer = ScheduleType_Sequential;

    auto& map_node_outer =
        builder.add_map(root, indvar_outer, condition_outer, init_outer, update_outer, schedule_type_outer);

    symbolic::Symbol indvar = symbolic::symbol("i");
    symbolic::Condition condition = symbolic::Lt(indvar, symbolic::integer(10));
    symbolic::Expression init = symbolic::zero();
    symbolic::Expression update = symbolic::add(indvar, symbolic::one());
    ScheduleType schedule_type = ScheduleType_Sequential;

    auto& map_node = builder.add_map(map_node_outer.root(), indvar, condition, init, update, schedule_type);

    auto sdfg = builder.move();

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& work_depth_analysis = analysis_manager.get<analysis::DegreesOfKnowledgeAnalysis>();

    auto& root_node = builder_opt.subject().root();
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
    ScheduleType schedule_type_outer = ScheduleType_Sequential;

    auto& map_node_outer =
        builder.add_map(root, indvar_outer, condition_outer, init_outer, update_outer, schedule_type_outer);

    symbolic::Symbol indvar = symbolic::symbol("i");
    symbolic::Condition condition = symbolic::Lt(indvar, symbolic::integer(10));
    symbolic::Expression init = symbolic::zero();
    symbolic::Expression update = symbolic::add(indvar, symbolic::one());
    ScheduleType schedule_type = ScheduleType_Sequential;

    auto& map_node = builder.add_map(map_node_outer.root(), indvar, condition, init, update, schedule_type);

    auto sdfg = builder.move();

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& work_depth_analysis = analysis_manager.get<analysis::DegreesOfKnowledgeAnalysis>();

    auto& root_node = builder_opt.subject().root();
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
    ScheduleType schedule_type_outer = ScheduleType_Sequential;

    auto& map_node_outer =
        builder.add_map(root, indvar_outer, condition_outer, init_outer, update_outer, schedule_type_outer);

    symbolic::Symbol indvar = symbolic::symbol("i");
    symbolic::Condition condition = symbolic::Lt(indvar, symbolic::integer(10));
    symbolic::Expression init = symbolic::zero();
    symbolic::Expression update = symbolic::add(indvar, symbolic::one());
    ScheduleType schedule_type = ScheduleType_Sequential;

    auto& map_node = builder.add_map(map_node_outer.root(), indvar, condition, init, update, schedule_type);

    auto sdfg = builder.move();

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& work_depth_analysis = analysis_manager.get<analysis::DegreesOfKnowledgeAnalysis>();

    auto& root_node = builder_opt.subject().root();
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
    ScheduleType schedule_type_outer = ScheduleType_Sequential;

    auto& map_node_outer =
        builder.add_map(root, indvar_outer, condition_outer, init_outer, update_outer, schedule_type_outer);

    symbolic::Symbol indvar = symbolic::symbol("i");
    symbolic::Condition condition = symbolic::Lt(indvar, symbolic::integer(10));
    symbolic::Expression init = symbolic::zero();
    symbolic::Expression update = symbolic::add(indvar, symbolic::one());
    ScheduleType schedule_type = ScheduleType_Sequential;

    auto& map_node = builder.add_map(map_node_outer.root(), indvar, condition, init, update, schedule_type);

    auto sdfg = builder.move();

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& work_depth_analysis = analysis_manager.get<analysis::DegreesOfKnowledgeAnalysis>();

    auto& root_node = builder_opt.subject().root();
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
    ScheduleType schedule_type = ScheduleType_Sequential;

    auto& map_node = builder.add_map(while_node.root(), indvar, condition, init, update, schedule_type);

    std::string while_symbol_name = "while_" + std::to_string(while_node.element_id());

    auto sdfg = builder.move();

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& work_depth_analysis = analysis_manager.get<analysis::DegreesOfKnowledgeAnalysis>();

    auto& root_node = builder_opt.subject().root();
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
    ScheduleType schedule_type = ScheduleType_Sequential;

    auto& map_node = builder.add_map(root, indvar, condition, init, update, schedule_type);

    auto& block = builder.add_block(map_node.root());
    auto& access_A = builder.add_access(block, "A");
    auto& access_B = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", sc_type}, {{"_in", sc_type}});

    // TODO: Add memlets

    auto sdfg = builder.move();

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& work_depth_analysis = analysis_manager.get<analysis::DegreesOfKnowledgeAnalysis>();

    auto& root_node = builder_opt.subject().root();
    auto work = work_depth_analysis.load_of_a_map(map_node);
    EXPECT_EQ(work.second, analysis::DegreesOfKnowledgeClassification::Scalar);
    EXPECT_TRUE(symbolic::eq(work.first, symbolic::integer(10)));
}
