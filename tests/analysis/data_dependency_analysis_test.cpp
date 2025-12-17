#include "sdfg/analysis/data_dependency_analysis.h"

#include <gtest/gtest.h>

#include <unordered_map>
#include <unordered_set>

#include "sdfg/analysis/users.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/element.h"
#include "sdfg/symbolic/symbolic.h"

using namespace sdfg;

TEST(DataDependencyAnalysisTest, Block_Define_Scalar) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    builder.add_container("A", base_desc);

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& output_node = builder.add_access(block, "A");
    auto& input_node = builder.add_constant(block, "0", base_desc);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {});
    builder.add_computational_memlet(block, input_node, tasklet, "_in", {});

    // Run analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    analysis::DataDependencyAnalysis analysis(builder.subject(), true);
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;
    analysis.visit_block(analysis_manager, block, undefined, open_definitions, closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 0);
    EXPECT_EQ(open_definitions.size(), 1);
    EXPECT_EQ(closed_definitions.size(), 0);

    auto& write_A = *open_definitions.begin();
    EXPECT_EQ(write_A.first->use(), analysis::Use::WRITE);
    EXPECT_EQ(write_A.first->container(), "A");
    EXPECT_EQ(write_A.first->element(), &output_node);
    EXPECT_EQ(write_A.second.size(), 0);
}

TEST(DataDependencyAnalysisTest, Block_Define_Array) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Array array_desc(base_desc, symbolic::integer(2));
    builder.add_container("A", array_desc);

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& output_node = builder.add_access(block, "A");
    auto& input_node = builder.add_constant(block, "0", base_desc);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {symbolic::integer(0)});
    builder.add_computational_memlet(block, input_node, tasklet, "_in", {});

    // Run analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    analysis::DataDependencyAnalysis analysis(builder.subject(), true);
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;
    analysis.visit_block(analysis_manager, block, undefined, open_definitions, closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 0);
    EXPECT_EQ(open_definitions.size(), 1);
    EXPECT_EQ(closed_definitions.size(), 0);

    auto& write_A = *open_definitions.begin();
    EXPECT_EQ(write_A.first->use(), analysis::Use::WRITE);
    EXPECT_EQ(write_A.first->container(), "A");
    EXPECT_EQ(write_A.first->element(), &output_node);
    EXPECT_EQ(write_A.second.size(), 0);
}

TEST(DataDependencyAnalysisTest, Block_Undefined_Scalar) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    builder.add_container("A", base_desc);
    builder.add_container("B", base_desc);

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& input_node = builder.add_access(block, "A");
    auto& output_node = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {});
    builder.add_computational_memlet(block, input_node, tasklet, "_in", {});

    // Run analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    analysis::DataDependencyAnalysis analysis(builder.subject(), true);
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;
    analysis.visit_block(analysis_manager, block, undefined, open_definitions, closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 1);
    EXPECT_EQ(open_definitions.size(), 1);
    EXPECT_EQ(closed_definitions.size(), 0);

    auto& write_B = *open_definitions.begin();
    EXPECT_EQ(write_B.first->use(), analysis::Use::WRITE);
    EXPECT_EQ(write_B.first->container(), "B");
    EXPECT_EQ(write_B.first->element(), &output_node);
    EXPECT_EQ(write_B.second.size(), 0);

    auto& read_A = *undefined.begin();
    EXPECT_EQ(read_A->use(), analysis::Use::READ);
    EXPECT_EQ(read_A->container(), "A");
    EXPECT_EQ(read_A->element(), &input_node);
}

TEST(DataDependencyAnalysisTest, Block_Undefined_Array) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Array array_desc(base_desc, symbolic::integer(2));
    builder.add_container("A", array_desc);
    builder.add_container("B", array_desc);

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& input_node = builder.add_access(block, "A");
    auto& output_node = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {symbolic::integer(0)});
    builder.add_computational_memlet(block, input_node, tasklet, "_in", {symbolic::integer(0)});

    // Run analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    analysis::DataDependencyAnalysis analysis(builder.subject(), true);
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;
    analysis.visit_block(analysis_manager, block, undefined, open_definitions, closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 1);
    EXPECT_EQ(open_definitions.size(), 1);
    EXPECT_EQ(closed_definitions.size(), 0);

    auto& write_B = *open_definitions.begin();
    EXPECT_EQ(write_B.first->use(), analysis::Use::WRITE);
    EXPECT_EQ(write_B.first->container(), "B");
    EXPECT_EQ(write_B.first->element(), &output_node);
    EXPECT_EQ(write_B.second.size(), 0);

    auto& read_A = *undefined.begin();
    EXPECT_EQ(read_A->use(), analysis::Use::READ);
    EXPECT_EQ(read_A->container(), "A");
    EXPECT_EQ(read_A->element(), &input_node);
}

TEST(DataDependencyAnalysisTest, Block_Undefined_Array_Subset) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Array array_desc(base_desc, symbolic::integer(2));
    builder.add_container("A", array_desc);
    builder.add_container("B", array_desc);
    builder.add_container("C", array_desc);

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& input_node = builder.add_access(block, "A");
    auto& output_node = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {symbolic::integer(0)});
    builder.add_computational_memlet(block, input_node, tasklet, "_in", {symbolic::integer(0)});

    auto& output_node2 = builder.add_access(block, "C");
    auto& tasklet2 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, tasklet2, "_out", output_node2, {symbolic::integer(0)});
    builder.add_computational_memlet(block, output_node, tasklet2, "_in", {symbolic::integer(1)});

    // Run analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    analysis::DataDependencyAnalysis analysis(builder.subject(), true);
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;
    analysis.visit_block(analysis_manager, block, undefined, open_definitions, closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 2);
    EXPECT_EQ(open_definitions.size(), 2);
    EXPECT_EQ(closed_definitions.size(), 0);

    auto read_A = users.get_user("A", &input_node, analysis::Use::READ);
    auto undefined_A = undefined.find(read_A);
    EXPECT_NE(undefined_A, undefined.end());

    auto read_B = users.get_user("B", &output_node, analysis::Use::READ);
    auto undefined_B = undefined.find(read_B);
    EXPECT_NE(undefined_B, undefined.end());

    auto write_B = users.get_user("B", &output_node, analysis::Use::WRITE);
    auto& definition_B = open_definitions.at(write_B);
    EXPECT_EQ(definition_B.size(), 0);

    auto write_C = users.get_user("C", &output_node2, analysis::Use::WRITE);
    auto& definition_C = open_definitions.at(write_C);
    EXPECT_EQ(definition_C.size(), 0);
}

TEST(DataDependencyAnalysisTest, Block_Undefined_Symbol) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Array array_desc(base_desc, symbolic::integer(2));
    builder.add_container("A", array_desc);
    builder.add_container("i", base_desc);

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& output_node = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"0"});
    auto& memlet = builder.add_computational_memlet(block, tasklet, "_out", output_node, {symbolic::symbol("i")});

    // Run analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    analysis::DataDependencyAnalysis analysis(builder.subject(), true);
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;
    analysis.visit_block(analysis_manager, block, undefined, open_definitions, closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 1);
    EXPECT_EQ(open_definitions.size(), 1);
    EXPECT_EQ(closed_definitions.size(), 0);

    auto& read_i = *undefined.begin();
    EXPECT_EQ(read_i->use(), analysis::Use::READ);
    EXPECT_EQ(read_i->container(), "i");
    EXPECT_EQ(read_i->element(), &memlet);

    auto& write_A = *open_definitions.begin();
    EXPECT_EQ(write_A.first->use(), analysis::Use::WRITE);
    EXPECT_EQ(write_A.first->container(), "A");
    EXPECT_EQ(write_A.first->element(), &output_node);
    EXPECT_EQ(write_A.second.size(), 0);
}

TEST(DataDependencyAnalysisTest, Block_Use_Scalar) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    builder.add_container("A", base_desc);
    builder.add_container("B", base_desc);
    builder.add_container("C", base_desc);

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& input_node = builder.add_access(block, "A");
    auto& output_node = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {});
    builder.add_computational_memlet(block, input_node, tasklet, "_in", {});

    auto& output_node2 = builder.add_access(block, "C");
    auto& tasklet2 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, tasklet2, "_out", output_node2, {});
    builder.add_computational_memlet(block, output_node, tasklet2, "_in", {});

    // Run analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    analysis::DataDependencyAnalysis analysis(builder.subject(), true);
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;
    analysis.visit_block(analysis_manager, block, undefined, open_definitions, closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 1);
    EXPECT_EQ(open_definitions.size(), 2);
    EXPECT_EQ(closed_definitions.size(), 0);

    auto& read_A = *undefined.begin();
    EXPECT_EQ(read_A->use(), analysis::Use::READ);
    EXPECT_EQ(read_A->container(), "A");
    EXPECT_EQ(read_A->element(), &input_node);

    auto write_B = users.get_user("B", &output_node, analysis::Use::WRITE);
    auto& definition_B = open_definitions.at(write_B);
    EXPECT_EQ(definition_B.size(), 1);
    EXPECT_EQ((*definition_B.begin())->use(), analysis::Use::READ);
    EXPECT_EQ((*definition_B.begin())->container(), "B");
    EXPECT_EQ((*definition_B.begin())->element(), &output_node);

    auto write_C = users.get_user("C", &output_node2, analysis::Use::WRITE);
    auto& definition_C = open_definitions.at(write_C);
    EXPECT_EQ(definition_C.size(), 0);
}

TEST(DataDependencyAnalysisTest, Block_Use_Array) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Array array_desc(base_desc, symbolic::integer(2));
    builder.add_container("A", array_desc);
    builder.add_container("B", array_desc);
    builder.add_container("C", array_desc);

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& input_node = builder.add_access(block, "A");
    auto& output_node = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {symbolic::integer(0)});
    builder.add_computational_memlet(block, input_node, tasklet, "_in", {symbolic::integer(0)});

    auto& output_node2 = builder.add_access(block, "C");
    auto& tasklet2 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, tasklet2, "_out", output_node2, {symbolic::integer(0)});
    builder.add_computational_memlet(block, output_node, tasklet2, "_in", {symbolic::integer(0)});

    // Run analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    analysis::DataDependencyAnalysis analysis(builder.subject(), true);
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;
    analysis.visit_block(analysis_manager, block, undefined, open_definitions, closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 1);
    EXPECT_EQ(open_definitions.size(), 2);
    EXPECT_EQ(closed_definitions.size(), 0);

    auto& read_A = *undefined.begin();
    EXPECT_EQ(read_A->use(), analysis::Use::READ);
    EXPECT_EQ(read_A->container(), "A");
    EXPECT_EQ(read_A->element(), &input_node);

    auto write_B = users.get_user("B", &output_node, analysis::Use::WRITE);
    auto& definition_B = open_definitions.at(write_B);
    EXPECT_EQ(definition_B.size(), 1);
    EXPECT_EQ((*definition_B.begin())->use(), analysis::Use::READ);
    EXPECT_EQ((*definition_B.begin())->container(), "B");
    EXPECT_EQ((*definition_B.begin())->element(), &output_node);

    auto write_C = users.get_user("C", &output_node2, analysis::Use::WRITE);
    auto& definition_C = open_definitions.at(write_C);
    EXPECT_EQ(definition_C.size(), 0);
}

TEST(DataDependencyAnalysisTest, Sequence_Define_Scalar) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    builder.add_container("A", base_desc);

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root, {{symbolic::symbol("A"), symbolic::integer(0)}});

    // Run analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    analysis::DataDependencyAnalysis analysis(builder.subject(), true);
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;
    analysis.visit_sequence(analysis_manager, root, undefined, open_definitions, closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 0);
    EXPECT_EQ(open_definitions.size(), 1);
    EXPECT_EQ(closed_definitions.size(), 0);

    auto& transition1 = root.at(0).second;

    auto& write_A = *open_definitions.begin();
    EXPECT_EQ(write_A.first->use(), analysis::Use::WRITE);
    EXPECT_EQ(write_A.first->container(), "A");
    EXPECT_EQ(write_A.first->element(), &transition1);
    EXPECT_EQ(write_A.second.size(), 0);
}

TEST(DataDependencyAnalysisTest, Sequence_Use_Scalar) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    builder.add_container("A", base_desc);
    builder.add_container("B", base_desc);

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root, {{symbolic::symbol("A"), symbolic::integer(0)}});
    auto& block2 = builder.add_block(root, {{symbolic::symbol("B"), symbolic::symbol("A")}});

    // Run analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    analysis::DataDependencyAnalysis analysis(builder.subject(), true);
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;
    analysis.visit_sequence(analysis_manager, root, undefined, open_definitions, closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 0);
    EXPECT_EQ(open_definitions.size(), 2);
    EXPECT_EQ(closed_definitions.size(), 0);

    auto& transition1 = root.at(0).second;
    auto& transition2 = root.at(1).second;

    auto write_A = users.get_user("A", &transition1, analysis::Use::WRITE);
    auto write_B = users.get_user("B", &transition2, analysis::Use::WRITE);

    auto& definition_A = open_definitions.at(write_A);
    EXPECT_EQ(definition_A.size(), 1);
    EXPECT_EQ((*definition_A.begin())->use(), analysis::Use::READ);
    EXPECT_EQ((*definition_A.begin())->container(), "A");
    EXPECT_EQ((*definition_A.begin())->element(), &transition2);

    auto& definition_B = open_definitions.at(write_B);
    EXPECT_EQ(definition_B.size(), 0);
}

TEST(DataDependencyAnalysisTest, Sequence_Close_Scalar) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    builder.add_container("A", base_desc);

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root, {{symbolic::symbol("A"), symbolic::integer(0)}});
    auto& block2 = builder.add_block(root, {{symbolic::symbol("A"), symbolic::integer(1)}});

    // Run analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    analysis::DataDependencyAnalysis analysis(builder.subject(), true);
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;
    analysis.visit_sequence(analysis_manager, root, undefined, open_definitions, closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 0);
    EXPECT_EQ(open_definitions.size(), 1);
    EXPECT_EQ(closed_definitions.size(), 1);

    auto& transition1 = root.at(0).second;
    auto& transition2 = root.at(1).second;

    auto write_A_1 = users.get_user("A", &transition1, analysis::Use::WRITE);
    auto write_A_2 = users.get_user("A", &transition2, analysis::Use::WRITE);

    auto& definition_A_1 = closed_definitions.at(write_A_1);
    EXPECT_EQ(definition_A_1.size(), 0);

    auto& definition_A_2 = open_definitions.at(write_A_2);
    EXPECT_EQ(definition_A_2.size(), 0);
}

TEST(DataDependencyAnalysisTest, Sequence_Close_Array) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Array array_desc(base_desc, symbolic::integer(2));
    builder.add_container("A", array_desc);

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root);
    auto& output_node = builder.add_access(block1, "A");
    auto& zero_node = builder.add_constant(block1, "0", base_desc);
    auto& tasklet = builder.add_tasklet(block1, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block1, zero_node, tasklet, "_in", {});
    builder.add_computational_memlet(block1, tasklet, "_out", output_node, {symbolic::integer(0)});

    auto& block2 = builder.add_block(root);
    auto& one_node = builder.add_constant(block2, "1", base_desc);
    auto& output_node2 = builder.add_access(block2, "A");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block2, one_node, tasklet2, "_in", {});
    builder.add_computational_memlet(block2, tasklet2, "_out", output_node2, {symbolic::integer(0)});

    // Run analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    analysis::DataDependencyAnalysis analysis(builder.subject(), true);
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;
    analysis.visit_sequence(analysis_manager, root, undefined, open_definitions, closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 0);
    EXPECT_EQ(open_definitions.size(), 1);
    EXPECT_EQ(closed_definitions.size(), 1);

    auto& transition1 = root.at(0).second;
    auto& transition2 = root.at(1).second;

    auto write_A_1 = users.get_user("A", &output_node, analysis::Use::WRITE);
    auto write_A_2 = users.get_user("A", &output_node2, analysis::Use::WRITE);

    auto& definition_A_1 = closed_definitions.at(write_A_1);
    EXPECT_EQ(definition_A_1.size(), 0);

    auto& definition_A_2 = open_definitions.at(write_A_2);
    EXPECT_EQ(definition_A_2.size(), 0);
}

TEST(DataDependencyAnalysisTest, Sequence_Define_Array_Subsets) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Array array_desc(base_desc, symbolic::integer(2));
    builder.add_container("A", array_desc);

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root);
    auto& output_node = builder.add_access(block1, "A");
    auto& tasklet = builder.add_tasklet(block1, data_flow::TaskletCode::assign, "_out", {"0"});
    builder.add_computational_memlet(block1, tasklet, "_out", output_node, {symbolic::integer(0)});

    auto& block2 = builder.add_block(root);
    auto& output_node2 = builder.add_access(block2, "A");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"1"});
    builder.add_computational_memlet(block2, tasklet2, "_out", output_node2, {symbolic::integer(1)});

    // Run analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    analysis::DataDependencyAnalysis analysis(builder.subject(), true);
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;
    analysis.visit_sequence(analysis_manager, root, undefined, open_definitions, closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 0);
    EXPECT_EQ(open_definitions.size(), 2);
    EXPECT_EQ(closed_definitions.size(), 0);

    auto& transition1 = root.at(0).second;
    auto& transition2 = root.at(1).second;

    auto write_A_1 = users.get_user("A", &output_node, analysis::Use::WRITE);
    auto write_A_2 = users.get_user("A", &output_node2, analysis::Use::WRITE);

    auto& definition_A_1 = open_definitions.at(write_A_1);
    EXPECT_EQ(definition_A_1.size(), 0);

    auto& definition_A_2 = open_definitions.at(write_A_2);
    EXPECT_EQ(definition_A_2.size(), 0);
}

TEST(DataDependencyAnalysisTest, IfElse_Condition_Undefined) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    builder.add_container("A", base_desc);

    auto& root = builder.subject().root();
    auto& if_else = builder.add_if_else(root);
    auto& branch1 = builder.add_case(if_else, symbolic::Eq(symbolic::symbol("A"), symbolic::integer(0)));

    // Run analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    analysis::DataDependencyAnalysis analysis(builder.subject(), true);
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;
    analysis.visit_sequence(analysis_manager, root, undefined, open_definitions, closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 1);
    EXPECT_EQ(open_definitions.size(), 0);
    EXPECT_EQ(closed_definitions.size(), 0);

    auto read_A = users.get_user("A", &if_else, analysis::Use::READ);
    auto undefined_A = undefined.find(read_A);
    EXPECT_NE(undefined_A, undefined.end());
}

TEST(DataDependencyAnalysisTest, IfElse_Condition_Use) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    builder.add_container("A", base_desc);

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root, {{symbolic::symbol("A"), symbolic::integer(0)}});
    auto& if_else = builder.add_if_else(root);
    auto& branch1 = builder.add_case(if_else, symbolic::Eq(symbolic::symbol("A"), symbolic::integer(0)));

    // Run analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    analysis::DataDependencyAnalysis analysis(builder.subject(), true);
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;
    analysis.visit_sequence(analysis_manager, root, undefined, open_definitions, closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 0);
    EXPECT_EQ(open_definitions.size(), 1);
    EXPECT_EQ(closed_definitions.size(), 0);

    auto& transition1 = root.at(0).second;

    auto write_A = users.get_user("A", &transition1, analysis::Use::WRITE);
    auto& definition_A = open_definitions.at(write_A);
    EXPECT_EQ(definition_A.size(), 1);
    EXPECT_EQ((*definition_A.begin())->use(), analysis::Use::READ);
    EXPECT_EQ((*definition_A.begin())->container(), "A");
    EXPECT_EQ((*definition_A.begin())->element(), &if_else);
}

TEST(DataDependencyAnalysisTest, IfElse_Define_Complete_Scalar) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    builder.add_container("A", base_desc);

    auto& root = builder.subject().root();
    auto& block0 = builder.add_block(root, {{symbolic::symbol("A"), symbolic::integer(0)}});
    auto& if_else = builder.add_if_else(root);
    auto& branch1 = builder.add_case(if_else, symbolic::__true__());
    auto& block1 = builder.add_block(branch1, {{symbolic::symbol("A"), symbolic::integer(1)}});
    auto& branch2 = builder.add_case(if_else, symbolic::__false__());
    auto& block2 = builder.add_block(branch2, {{symbolic::symbol("A"), symbolic::integer(2)}});
    auto& block3 = builder.add_block(root, {{symbolic::symbol("B"), symbolic::symbol("A")}});

    // Run analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    analysis::DataDependencyAnalysis analysis(builder.subject(), true);
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;
    analysis.visit_sequence(analysis_manager, root, undefined, open_definitions, closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 0);
    EXPECT_EQ(open_definitions.size(), 3);
    EXPECT_EQ(closed_definitions.size(), 1);

    auto& transition0 = root.at(0).second;
    auto& transition1 = branch1.at(0).second;
    auto& transition2 = branch2.at(0).second;
    auto& transition3 = root.at(2).second;

    auto write_A_0 = users.get_user("A", &transition0, analysis::Use::WRITE);
    auto write_A_1 = users.get_user("A", &transition1, analysis::Use::WRITE);
    auto write_A_2 = users.get_user("A", &transition2, analysis::Use::WRITE);
    auto write_B = users.get_user("B", &transition3, analysis::Use::WRITE);
    auto read_A = users.get_user("A", &transition3, analysis::Use::READ);

    auto& definition_A_0 = closed_definitions.at(write_A_0);
    EXPECT_EQ(definition_A_0.size(), 0);

    auto& definition_A_1 = open_definitions.at(write_A_1);
    EXPECT_EQ(definition_A_1.size(), 1);
    EXPECT_NE(definition_A_1.find(read_A), definition_A_1.end());

    auto& definition_A_2 = open_definitions.at(write_A_2);
    EXPECT_EQ(definition_A_2.size(), 1);
    EXPECT_NE(definition_A_2.find(read_A), definition_A_2.end());

    auto& definition_B = open_definitions.at(write_B);
    EXPECT_EQ(definition_B.size(), 0);
}

TEST(DataDependencyAnalysisTest, IfElse_Define_Incomplete_Scalar) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    builder.add_container("A", base_desc);
    builder.add_container("B", base_desc);

    auto& root = builder.subject().root();
    auto& block0 = builder.add_block(root, {{symbolic::symbol("A"), symbolic::integer(0)}});
    auto& if_else = builder.add_if_else(root);
    auto& branch1 = builder.add_case(if_else, symbolic::__false__());
    auto& block1 = builder.add_block(branch1, {{symbolic::symbol("A"), symbolic::integer(1)}});
    auto& block2 = builder.add_block(root, {{symbolic::symbol("B"), symbolic::symbol("A")}});

    // Run analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    analysis::DataDependencyAnalysis analysis(builder.subject(), true);
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;
    analysis.visit_sequence(analysis_manager, root, undefined, open_definitions, closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 0);
    EXPECT_EQ(open_definitions.size(), 4);
    EXPECT_EQ(closed_definitions.size(), 0);

    auto& transition0 = root.at(0).second;
    auto& transition1 = branch1.at(0).second;
    auto& transition3 = root.at(2).second;

    auto write_A_0 = users.get_user("A", &transition0, analysis::Use::WRITE);
    auto write_A_1 = users.get_user("A", &transition1, analysis::Use::WRITE);
    auto write_B = users.get_user("B", &transition3, analysis::Use::WRITE);
    auto read_A = users.get_user("A", &transition3, analysis::Use::READ);

    auto& definition_A_0 = open_definitions.at(write_A_0);
    EXPECT_EQ(definition_A_0.size(), 1);
    EXPECT_NE(definition_A_0.find(read_A), definition_A_0.end());

    auto& definition_A_1 = open_definitions.at(write_A_1);
    EXPECT_EQ(definition_A_1.size(), 1);
    EXPECT_NE(definition_A_1.find(read_A), definition_A_1.end());

    auto& definition_B = open_definitions.at(write_B);
    EXPECT_EQ(definition_B.size(), 0);
}

TEST(DataDependencyAnalysisTest, IfElse_Define_Array) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Array array_desc(base_desc, symbolic::integer(2));
    builder.add_container("A", array_desc);

    auto& root = builder.subject().root();
    auto& if_else = builder.add_if_else(root);

    auto& branch1 = builder.add_case(if_else, symbolic::__true__());
    auto& block1 = builder.add_block(branch1);
    auto& output_node = builder.add_access(block1, "A");
    auto& tasklet = builder.add_tasklet(block1, data_flow::TaskletCode::assign, "_out", {"0"});
    builder.add_computational_memlet(block1, tasklet, "_out", output_node, {symbolic::integer(0)});

    auto& branch2 = builder.add_case(if_else, symbolic::__false__());
    auto& block2 = builder.add_block(branch2);
    auto& output_node2 = builder.add_access(block2, "A");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"1"});
    builder.add_computational_memlet(block2, tasklet2, "_out", output_node2, {symbolic::integer(0)});

    auto& block3 = builder.add_block(root);
    auto& output_node3 = builder.add_access(block3, "A");
    auto& tasklet3 = builder.add_tasklet(block3, data_flow::TaskletCode::assign, "_out", {"0"});
    builder.add_computational_memlet(block3, tasklet3, "_out", output_node3, {symbolic::integer(0)});

    // Run analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    analysis::DataDependencyAnalysis analysis(builder.subject(), true);
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;
    analysis.visit_sequence(analysis_manager, root, undefined, open_definitions, closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 0);
    EXPECT_EQ(open_definitions.size(), 1);
    EXPECT_EQ(closed_definitions.size(), 2);

    auto& transition1 = branch1.at(0).second;
    auto& transition2 = branch2.at(0).second;
    auto& transition3 = root.at(1).second;

    auto write_A_1 = users.get_user("A", &output_node, analysis::Use::WRITE);
    auto write_A_2 = users.get_user("A", &output_node2, analysis::Use::WRITE);
    auto write_A_3 = users.get_user("A", &output_node3, analysis::Use::WRITE);

    auto& definition_A_1 = closed_definitions.at(write_A_1);
    EXPECT_EQ(definition_A_1.size(), 0);

    auto& definition_A_2 = closed_definitions.at(write_A_2);
    EXPECT_EQ(definition_A_2.size(), 0);

    auto& definition_A_3 = open_definitions.at(write_A_3);
    EXPECT_EQ(definition_A_3.size(), 0);
}

TEST(DataDependencyAnalysisTest, IfElse_Define_Array_Subsets) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Array array_desc(base_desc, symbolic::integer(2));
    builder.add_container("A", array_desc);

    auto& root = builder.subject().root();
    auto& if_else = builder.add_if_else(root);

    auto& branch1 = builder.add_case(if_else, symbolic::__true__());
    auto& block1 = builder.add_block(branch1);
    auto& output_node = builder.add_access(block1, "A");
    auto& tasklet = builder.add_tasklet(block1, data_flow::TaskletCode::assign, "_out", {"0"});
    builder.add_computational_memlet(block1, tasklet, "_out", output_node, {symbolic::integer(0)});

    auto& branch2 = builder.add_case(if_else, symbolic::__false__());
    auto& block2 = builder.add_block(branch2);
    auto& output_node2 = builder.add_access(block2, "A");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"1"});
    builder.add_computational_memlet(block2, tasklet2, "_out", output_node2, {symbolic::integer(0)});

    auto& block3 = builder.add_block(root);
    auto& output_node3 = builder.add_access(block3, "A");
    auto& tasklet3 = builder.add_tasklet(block3, data_flow::TaskletCode::assign, "_out", {"0"});
    builder.add_computational_memlet(block3, tasklet3, "_out", output_node3, {symbolic::integer(1)});

    // Run analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    analysis::DataDependencyAnalysis analysis(builder.subject(), true);
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;
    analysis.visit_sequence(analysis_manager, root, undefined, open_definitions, closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 0);
    EXPECT_EQ(open_definitions.size(), 3);
    EXPECT_EQ(closed_definitions.size(), 0);

    auto& transition1 = branch1.at(0).second;
    auto& transition2 = branch2.at(0).second;
    auto& transition3 = root.at(1).second;

    auto write_A_1 = users.get_user("A", &output_node, analysis::Use::WRITE);
    auto write_A_2 = users.get_user("A", &output_node2, analysis::Use::WRITE);
    auto write_A_3 = users.get_user("A", &output_node3, analysis::Use::WRITE);

    auto& definition_A_1 = open_definitions.at(write_A_1);
    EXPECT_EQ(definition_A_1.size(), 0);

    auto& definition_A_2 = open_definitions.at(write_A_2);
    EXPECT_EQ(definition_A_2.size(), 0);

    auto& definition_A_3 = open_definitions.at(write_A_3);
    EXPECT_EQ(definition_A_3.size(), 0);
}

TEST(DataDependencyAnalysisTest, IfElse_Close_Array) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Array array_desc(base_desc, symbolic::integer(2));
    builder.add_container("A", array_desc);

    auto& root = builder.subject().root();
    auto& if_else = builder.add_if_else(root);

    auto& branch1 = builder.add_case(if_else, symbolic::__true__());
    auto& block1 = builder.add_block(branch1);
    auto& output_node = builder.add_access(block1, "A");
    auto& tasklet = builder.add_tasklet(block1, data_flow::TaskletCode::assign, "_out", {"0"});
    builder.add_computational_memlet(block1, tasklet, "_out", output_node, {symbolic::integer(1)}, array_desc);

    auto& branch2 = builder.add_case(if_else, symbolic::__false__());
    auto& block2 = builder.add_block(branch2);
    auto& output_node2 = builder.add_access(block2, "A");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"1"});
    builder.add_computational_memlet(block2, tasklet2, "_out", output_node2, {symbolic::integer(0)}, array_desc);

    auto& block3 = builder.add_block(root);
    auto& output_node3 = builder.add_access(block3, "A");
    auto& tasklet3 = builder.add_tasklet(block3, data_flow::TaskletCode::assign, "_out", {"0"});
    builder.add_computational_memlet(block3, tasklet3, "_out", output_node3, {symbolic::integer(1)}, array_desc);

    // Run analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    analysis::DataDependencyAnalysis analysis(builder.subject(), true);
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;
    analysis.visit_sequence(analysis_manager, root, undefined, open_definitions, closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 0);
    EXPECT_EQ(open_definitions.size(), 2);
    EXPECT_EQ(closed_definitions.size(), 1);

    auto& transition1 = branch1.at(0).second;
    auto& transition2 = branch2.at(0).second;
    auto& transition3 = root.at(1).second;

    auto write_A_1 = users.get_user("A", &output_node, analysis::Use::WRITE);
    auto write_A_2 = users.get_user("A", &output_node2, analysis::Use::WRITE);
    auto write_A_3 = users.get_user("A", &output_node3, analysis::Use::WRITE);

    auto& definition_A_1 = closed_definitions.at(write_A_1);
    EXPECT_EQ(definition_A_1.size(), 0);

    auto& definition_A_2 = open_definitions.at(write_A_2);
    EXPECT_EQ(definition_A_2.size(), 0);

    auto& definition_A_3 = open_definitions.at(write_A_3);
    EXPECT_EQ(definition_A_3.size(), 0);
}

TEST(DataDependencyAnalysisTest, For_Indvar) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));

    auto& root = builder.subject().root();

    auto& for_loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );

    analysis::AnalysisManager analysis_manager(builder.subject());
    analysis::DataDependencyAnalysis analysis(builder.subject(), true);
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;
    analysis.visit_sequence(analysis_manager, root, undefined, open_definitions, closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 0);
    EXPECT_EQ(open_definitions.size(), 2);
    EXPECT_EQ(closed_definitions.size(), 0);

    auto init_i = users.get_user("i", &for_loop, analysis::Use::WRITE, true, false, false);
    auto update_i_write = users.get_user("i", &for_loop, analysis::Use::WRITE, false, false, true);
    auto update_i_read = users.get_user("i", &for_loop, analysis::Use::READ, false, false, true);
    auto condition_i = users.get_user("i", &for_loop, analysis::Use::READ, false, true, false);

    auto& definition_init_i = open_definitions.at(init_i);
    EXPECT_EQ(definition_init_i.size(), 2);
    EXPECT_NE(definition_init_i.find(update_i_read), definition_init_i.end());
    EXPECT_NE(definition_init_i.find(condition_i), definition_init_i.end());

    auto& definition_update_i_write = open_definitions.at(update_i_write);
    EXPECT_EQ(definition_update_i_write.size(), 2);
    EXPECT_NE(definition_update_i_write.find(update_i_read), definition_update_i_write.end());
    EXPECT_NE(definition_update_i_write.find(condition_i), definition_update_i_write.end());
}

TEST(DataDependencyAnalysisTest, For_Close_Scalar) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("A", types::Scalar(types::PrimitiveType::Int32));

    auto& root = builder.subject().root();

    auto& for_loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(1)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );

    auto& block = builder.add_block(for_loop.root());
    auto& output_node = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"0"});
    builder
        .add_computational_memlet(block, tasklet, "_out", output_node, {}, types::Scalar(types::PrimitiveType::Int32));

    auto& block2 = builder.add_block(root);
    auto& output_node2 = builder.add_access(block2, "A");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"0"});
    builder
        .add_computational_memlet(block2, tasklet2, "_out", output_node2, {}, types::Scalar(types::PrimitiveType::Int32));

    analysis::AnalysisManager analysis_manager(builder.subject());
    analysis::DataDependencyAnalysis analysis(builder.subject(), true);
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;
    analysis.visit_sequence(analysis_manager, root, undefined, open_definitions, closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 0);
    EXPECT_EQ(open_definitions.size(), 3);
    EXPECT_EQ(closed_definitions.size(), 1);

    auto write_A_body = users.get_user("A", &output_node, analysis::Use::WRITE);
    auto write_A_after = users.get_user("A", &output_node2, analysis::Use::WRITE);

    auto& definition_A_body = closed_definitions.at(write_A_body);
    EXPECT_EQ(definition_A_body.size(), 0);

    auto& definition_A_after = open_definitions.at(write_A_after);
    EXPECT_EQ(definition_A_after.size(), 0);
}

TEST(DataDependencyAnalysisTest, For_Close_Array) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Array array_desc(base_desc, symbolic::integer(2));
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("j", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("A", array_desc);

    auto& root = builder.subject().root();

    auto& for_loop1 = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(2)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );

    auto& block = builder.add_block(for_loop1.root());
    auto& output_node = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"0"});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {symbolic::symbol("i")}, array_desc);

    auto& for_loop2 = builder.add_for(
        root,
        symbolic::symbol("j"),
        symbolic::Lt(symbolic::symbol("j"), symbolic::integer(2)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1))
    );

    auto& block2 = builder.add_block(for_loop2.root());
    auto& output_node2 = builder.add_access(block2, "A");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"0"});
    builder.add_computational_memlet(block2, tasklet2, "_out", output_node2, {symbolic::symbol("j")}, array_desc);

    analysis::AnalysisManager analysis_manager(builder.subject());
    analysis::DataDependencyAnalysis analysis(builder.subject(), true);
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;
    analysis.visit_sequence(analysis_manager, root, undefined, open_definitions, closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 0);
    EXPECT_EQ(open_definitions.size(), 5);
    EXPECT_EQ(closed_definitions.size(), 1);

    auto write_A_body = users.get_user("A", &output_node, analysis::Use::WRITE);
    auto write_A_after = users.get_user("A", &output_node2, analysis::Use::WRITE);

    auto& definition_A_body = closed_definitions.at(write_A_body);
    EXPECT_EQ(definition_A_body.size(), 0);

    auto& definition_A_after = open_definitions.at(write_A_after);
    EXPECT_EQ(definition_A_after.size(), 0);
}

TEST(DataDependencyAnalysisTest, For_Close_Array_Subsets) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Array array_desc(base_desc, symbolic::integer(2));
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("j", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("A", array_desc);

    auto& root = builder.subject().root();

    auto& for_loop1 = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(2)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );

    auto& block = builder.add_block(for_loop1.root());
    auto& output_node = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"0"});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {symbolic::symbol("i")}, array_desc);

    auto& for_loop2 = builder.add_for(
        root,
        symbolic::symbol("j"),
        symbolic::Lt(symbolic::symbol("j"), symbolic::integer(1)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1))
    );

    auto& block2 = builder.add_block(for_loop2.root());
    auto& output_node2 = builder.add_access(block2, "A");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"0"});
    builder.add_computational_memlet(block2, tasklet2, "_out", output_node2, {symbolic::symbol("j")}, array_desc);

    analysis::AnalysisManager analysis_manager(builder.subject());
    analysis::DataDependencyAnalysis analysis(builder.subject(), true);
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;
    analysis.visit_sequence(analysis_manager, root, undefined, open_definitions, closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 0);
    EXPECT_EQ(open_definitions.size(), 6);
    EXPECT_EQ(closed_definitions.size(), 0);

    auto write_A_body = users.get_user("A", &output_node, analysis::Use::WRITE);
    auto write_A_after = users.get_user("A", &output_node2, analysis::Use::WRITE);

    auto& definition_A_body = open_definitions.at(write_A_body);
    EXPECT_EQ(definition_A_body.size(), 0);

    auto& definition_A_after = open_definitions.at(write_A_after);
    EXPECT_EQ(definition_A_after.size(), 0);
}

TEST(DataDependencyAnalysisTest, For_Close_Array_Subsets_Trivial) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Array array_desc(base_desc, symbolic::integer(2));
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("j", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("A", array_desc);

    auto& root = builder.subject().root();

    auto& for_loop1 = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );

    auto& block = builder.add_block(for_loop1.root());
    auto& output_node = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"0"});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {symbolic::integer(0)}, array_desc);

    auto& for_loop2 = builder.add_for(
        root,
        symbolic::symbol("j"),
        symbolic::Lt(symbolic::symbol("j"), symbolic::integer(1)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1))
    );

    auto& block2 = builder.add_block(for_loop2.root());
    auto& output_node2 = builder.add_access(block2, "A");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"0"});
    builder.add_computational_memlet(block2, tasklet2, "_out", output_node2, {symbolic::integer(0)}, array_desc);

    analysis::AnalysisManager analysis_manager(builder.subject());
    analysis::DataDependencyAnalysis analysis(builder.subject(), true);
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;
    analysis.visit_sequence(analysis_manager, root, undefined, open_definitions, closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 0);
    EXPECT_EQ(open_definitions.size(), 5);
    EXPECT_EQ(closed_definitions.size(), 1);

    auto write_A_body = users.get_user("A", &output_node, analysis::Use::WRITE);
    auto write_A_after = users.get_user("A", &output_node2, analysis::Use::WRITE);

    auto& definition_A_body = closed_definitions.at(write_A_body);
    EXPECT_EQ(definition_A_body.size(), 0);

    auto& definition_A_after = open_definitions.at(write_A_after);
    EXPECT_EQ(definition_A_after.size(), 0);
}

TEST(LoopDependencyAnalysisTest, Last_1D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("B", base_desc, true);

    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& a1 = builder.add_access(block, "A");
    auto& b_out = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a1, tasklet, "_in", {indvar}, edge_desc);
    builder.add_computational_memlet(block, tasklet, "_out", b_out, {});

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    analysis::DataDependencyAnalysis analysis(sdfg, true);
    analysis.run(analysis_manager);
    auto& dependencies = analysis.dependencies(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 1);
    EXPECT_EQ(dependencies.at("B"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
}

TEST(LoopDependencyAnalysisTest, Sum_1D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("B", base_desc, true);

    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& a1 = builder.add_access(block, "A");
    auto& b_in = builder.add_access(block, "B");
    auto& b_out = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, a1, tasklet, "_in1", {indvar}, edge_desc);
    builder.add_computational_memlet(block, b_in, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", b_out, {});

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    analysis::DataDependencyAnalysis analysis(sdfg, true);
    analysis.run(analysis_manager);
    auto& dependencies = analysis.dependencies(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 1);
    EXPECT_EQ(dependencies.at("B"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_READ_WRITE);
}

TEST(LoopDependencyAnalysisTest, Shift_1D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);

    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(1);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& a1 = builder.add_access(block, "A");
    auto& a2 = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder
        .add_computational_memlet(block, a1, tasklet, "_in", {symbolic::sub(indvar, symbolic::integer(1))}, edge_desc);
    builder.add_computational_memlet(block, tasklet, "_out", a2, {indvar}, edge_desc);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    analysis::DataDependencyAnalysis analysis(sdfg, true);
    analysis.run(analysis_manager);
    auto& dependencies = analysis.dependencies(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 1);
    EXPECT_EQ(dependencies.at("A"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_READ_WRITE);
}

TEST(LoopDependencyAnalysisTest, PartialSum_1D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(1);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& A1 = builder.add_access(block, "A");
    auto& A2 = builder.add_access(block, "A");
    auto& A3 = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder
        .add_computational_memlet(block, A1, tasklet, "_in1", {symbolic::sub(indvar, symbolic::integer(1))}, edge_desc);
    builder.add_computational_memlet(block, A2, tasklet, "_in2", {indvar}, edge_desc);
    builder.add_computational_memlet(block, tasklet, "_out", A3, {indvar}, edge_desc);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    analysis::DataDependencyAnalysis analysis(sdfg, true);
    analysis.run(analysis_manager);
    auto& dependencies = analysis.dependencies(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 1);
    EXPECT_EQ(dependencies.at("A"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_READ_WRITE);
}

TEST(LoopDependencyAnalysisTest, LoopLocal_1D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("tmp", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block_1 = builder.add_block(body);
    auto& i_in = builder.add_access(block_1, "i");
    auto& tmp_out = builder.add_access(block_1, "tmp");
    auto& tasklet_1 = builder.add_tasklet(block_1, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block_1, i_in, tasklet_1, "_in", {});
    builder.add_computational_memlet(block_1, tasklet_1, "_out", tmp_out, {});

    auto& block_2 = builder.add_block(body);
    auto& tmp_in = builder.add_access(block_2, "tmp");
    auto& a_out = builder.add_access(block_2, "A");
    auto& tasklet = builder.add_tasklet(block_2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block_2, tmp_in, tasklet, "_in", {});
    builder.add_computational_memlet(block_2, tasklet, "_out", a_out, {indvar}, edge_desc);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    analysis::DataDependencyAnalysis analysis(sdfg, true);
    analysis.run(analysis_manager);
    auto& dependencies = analysis.dependencies(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 1);
    EXPECT_EQ(dependencies.at("tmp"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
}

TEST(LoopDependencyAnalysisTest, LoopLocal_Conditional) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("tmp", sym_desc);

    builder.add_container("A", desc, true);

    // Define loop
    auto bound1 = symbolic::symbol("N");
    auto indvar1 = symbolic::symbol("i");
    auto init1 = symbolic::integer(0);
    auto condition1 = symbolic::Lt(indvar1, bound1);
    auto update1 = symbolic::add(indvar1, symbolic::integer(1));

    auto& loop1 = builder.add_for(root, indvar1, condition1, init1, update1);
    auto& body1 = loop1.root();

    auto& ifelse = builder.add_if_else(body1);
    auto& branch1 = builder.add_case(ifelse, symbolic::Eq(indvar1, symbolic::integer(0)));
    auto& block1 = builder.add_block(branch1, {{symbolic::symbol("tmp"), symbolic::zero()}});
    auto& branch2 = builder.add_case(ifelse, symbolic::Ne(indvar1, symbolic::integer(0)));
    auto& block2 = builder.add_block(branch2, {{symbolic::symbol("tmp"), symbolic::one()}});

    // Add computation
    auto& block = builder.add_block(body1);
    auto& A_in = builder.add_access(block, "tmp");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, A_in, tasklet, "_in", {});
    builder.add_computational_memlet(block, tasklet, "_out", A_out, {indvar1}, edge_desc);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    analysis::DataDependencyAnalysis analysis(sdfg, true);
    analysis.run(analysis_manager);
    auto& dependencies = analysis.dependencies(loop1);

    // Check
    EXPECT_EQ(dependencies.size(), 1);
    EXPECT_EQ(dependencies.at("tmp"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
}

TEST(LoopDependencyAnalysisTest, LoopLocal_Conditional_Incomplete) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("tmp", sym_desc);
    builder.add_container("A", desc, true);

    // Define loop
    auto bound1 = symbolic::symbol("N");
    auto indvar1 = symbolic::symbol("i");
    auto init1 = symbolic::integer(0);
    auto condition1 = symbolic::Lt(indvar1, bound1);
    auto update1 = symbolic::add(indvar1, symbolic::integer(1));

    auto& loop1 = builder.add_for(root, indvar1, condition1, init1, update1);
    auto& body1 = loop1.root();

    auto& ifelse = builder.add_if_else(body1);
    auto& branch1 = builder.add_case(ifelse, symbolic::Eq(indvar1, symbolic::integer(0)));
    auto& block1 = builder.add_block(branch1, {{symbolic::symbol("tmp"), symbolic::zero()}});

    // Add computation
    auto& block = builder.add_block(body1);
    auto& A_in = builder.add_access(block, "tmp");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, A_in, tasklet, "_in", {});
    builder.add_computational_memlet(block, tasklet, "_out", A_out, {indvar1}, edge_desc);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    analysis::DataDependencyAnalysis analysis(sdfg, true);
    analysis.run(analysis_manager);
    auto& dependencies = analysis.dependencies(loop1);

    // Check
    EXPECT_EQ(dependencies.size(), 1);
    EXPECT_EQ(dependencies.at("tmp"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_READ_WRITE);
}

TEST(LoopDependencyAnalysisTest, Store_1D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& a = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"0"});
    builder.add_computational_memlet(block, tasklet, "_out", a, {indvar}, edge_desc);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    analysis::DataDependencyAnalysis analysis(sdfg, true);
    analysis.run(analysis_manager);
    auto& dependencies = analysis.dependencies(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 0);
}

TEST(LoopDependencyAnalysisTest, Copy_1D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("B", desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& a = builder.add_access(block, "A");
    auto& b = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a, tasklet, "_in", {symbolic::symbol("i")}, edge_desc);
    builder.add_computational_memlet(block, tasklet, "_out", b, {symbolic::symbol("i")}, edge_desc);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    analysis::DataDependencyAnalysis analysis(sdfg, true);
    analysis.run(analysis_manager);
    auto& dependencies = analysis.dependencies(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 0);
}

TEST(LoopDependencyAnalysisTest, Map_1D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& a_in = builder.add_access(block, "A");
    auto& one_node = builder.add_constant(block, "1.0", base_desc);
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in1", {symbolic::symbol("i")}, edge_desc);
    builder.add_computational_memlet(block, one_node, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", a_out, {symbolic::symbol("i")}, edge_desc);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    analysis::DataDependencyAnalysis analysis(sdfg, true);
    analysis.run(analysis_manager);
    auto& dependencies = analysis.dependencies(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 0);
}

TEST(LoopDependencyAnalysisTest, Map_1D_Disjoint) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& a_in = builder.add_access(block, "A");
    auto& one_node = builder.add_constant(block, "1.0", base_desc);
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in1", {symbolic::zero()}, edge_desc);
    builder.add_computational_memlet(block, one_node, tasklet, "_in2", {});
    builder.add_computational_memlet(
        block, tasklet, "_out", a_out, {symbolic::add(symbolic::symbol("i"), symbolic::one())}, edge_desc
    );

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    analysis::DataDependencyAnalysis analysis(sdfg, true);
    analysis.run(analysis_manager);
    auto& dependencies = analysis.dependencies(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 0);
}

TEST(LoopDependencyAnalysisTest, Map_1D_Strided) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::sub(symbolic::symbol("N"), symbolic::integer(1));
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(2));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& a_in = builder.add_access(block, "A");
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in", {symbolic::symbol("i")}, edge_desc);
    builder.add_computational_memlet(
        block, tasklet, "_out", a_out, {symbolic::add(symbolic::symbol("i"), symbolic::one())}, edge_desc
    );

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    analysis::DataDependencyAnalysis analysis(sdfg, true);
    analysis.run(analysis_manager);
    auto& dependencies = analysis.dependencies(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 0);
}

TEST(LoopDependencyAnalysisTest, Map_1D_Strided2) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(1);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(2));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& a_in = builder.add_access(block, "A");
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in", {symbolic::symbol("i")}, edge_desc);
    builder.add_computational_memlet(
        block, tasklet, "_out", a_out, {symbolic::sub(symbolic::symbol("i"), symbolic::one())}, edge_desc
    );

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    analysis::DataDependencyAnalysis analysis(sdfg, true);
    analysis.run(analysis_manager);
    auto& dependencies = analysis.dependencies(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 0);
}

TEST(LoopDependencyAnalysisTest, Map_1D_Tiled) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("i_tile", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i_tile");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto tile_size = symbolic::integer(8);
    auto update = symbolic::add(indvar, tile_size);

    auto& loop_outer = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop_outer.root();

    auto indvar_tile = symbolic::symbol("i");
    auto init_tile = indvar;
    auto condition_tile = symbolic::
        And(symbolic::Lt(indvar_tile, symbolic::symbol("N")),
            symbolic::Lt(indvar_tile, symbolic::add(indvar, tile_size)));
    auto update_tile = symbolic::add(indvar_tile, symbolic::one());

    auto& loop_inner = builder.add_for(body, indvar_tile, condition_tile, init_tile, update_tile);
    auto& body_inner = loop_inner.root();

    // Add computation
    auto& block = builder.add_block(body_inner);
    auto& a_in = builder.add_access(block, "A");
    auto& one_node = builder.add_constant(block, "1.0", base_desc);
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in1", {symbolic::symbol("i")}, edge_desc);
    builder.add_computational_memlet(block, one_node, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", a_out, {symbolic::symbol("i")}, edge_desc);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    analysis::DataDependencyAnalysis analysis(sdfg, true);
    analysis.run(analysis_manager);
    auto& dependencies1 = analysis.dependencies(loop_outer);
    auto& dependencies2 = analysis.dependencies(loop_inner);

    // Check
    EXPECT_EQ(dependencies1.size(), 1);
    EXPECT_EQ(dependencies1.at("i"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
    EXPECT_EQ(dependencies2.size(), 0);
}

TEST(LoopDependencyAnalysisTest, Map_1D_Incomplete) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar bool_desc(types::PrimitiveType::Bool);
    types::Scalar sym_desc(types::PrimitiveType::Int32);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("tmp", base_desc);
    builder.add_container("k", bool_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // tmp = A[i]
    auto& block = builder.add_block(body);
    auto& a_in = builder.add_access(block, "A");
    auto& tmp_out = builder.add_access(block, "tmp");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in", {indvar}, edge_desc);
    builder.add_computational_memlet(block, tasklet, "_out", tmp_out, {});

    // switch = tmp > 0
    auto& block_switch = builder.add_block(body);
    auto& tmp_in = builder.add_access(block_switch, "tmp");
    auto& zero_node = builder.add_constant(block_switch, "0.0", base_desc);
    auto& switch_out = builder.add_access(block_switch, "k");
    auto& tasklet_switch = builder.add_tasklet(block_switch, data_flow::TaskletCode::fp_oge, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block_switch, tmp_in, tasklet_switch, "_in1", {});
    builder.add_computational_memlet(block_switch, zero_node, tasklet_switch, "_in2", {});
    builder.add_computational_memlet(block_switch, tasklet_switch, "_out", switch_out, {});

    auto switch_condition = symbolic::Eq(symbolic::symbol("k"), symbolic::__true__());
    auto& ifelse = builder.add_if_else(body);

    // if (switch) A[i] = tmp
    auto& branch1 = builder.add_case(ifelse, switch_condition);
    auto& block1 = builder.add_block(branch1);
    auto& tmp_in1 = builder.add_access(block1, "tmp");
    auto& a_out1 = builder.add_access(block1, "A");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block1, tmp_in1, tasklet1, "_in", {});
    builder.add_computational_memlet(block1, tasklet1, "_out", a_out1, {indvar}, edge_desc);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    analysis::DataDependencyAnalysis analysis(sdfg, true);
    analysis.run(analysis_manager);
    auto& dependencies = analysis.dependencies(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 2);
    EXPECT_EQ(dependencies.at("tmp"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
    EXPECT_EQ(dependencies.at("k"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
}

TEST(LoopDependencyAnalysisTest, MapParameterized_1D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("m", sym_desc, true);
    builder.add_container("b", sym_desc, true);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in1"});
    builder.add_computational_memlet(
        block,
        A_in,
        tasklet,
        "_in1",
        {symbolic::add(symbolic::mul(symbolic::symbol("m"), symbolic::symbol("i")), symbolic::symbol("b"))},
        edge_desc
    );
    builder.add_computational_memlet(
        block,
        tasklet,
        "_out",
        A_out,
        {symbolic::add(symbolic::mul(symbolic::symbol("m"), symbolic::symbol("i")), symbolic::symbol("b"))},
        edge_desc
    );

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    analysis::DataDependencyAnalysis analysis(sdfg, true);
    analysis.run(analysis_manager);
    auto& dependencies = analysis.dependencies(loop);

    // Check
    // m == 0 -> all iterations access the same location
    EXPECT_EQ(dependencies.size(), 1);
    EXPECT_EQ(dependencies.at("A"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_READ_WRITE);
}

TEST(LoopDependencyAnalysisTest, Stencil_1D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("B", desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(1);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& A1 = builder.add_access(block, "A");
    auto& A2 = builder.add_access(block, "A");
    auto& A3 = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_fma, "_out", {"_in1", "_in2", "_in3"});
    builder.add_computational_memlet(
        block, A1, tasklet, "_in1", {symbolic::sub(symbolic::symbol("i"), symbolic::integer(1))}, edge_desc
    );
    builder.add_computational_memlet(block, A2, tasklet, "_in2", {symbolic::symbol("i")}, edge_desc);
    builder.add_computational_memlet(
        block, A3, tasklet, "_in3", {symbolic::add(symbolic::symbol("i"), symbolic::integer(1))}, edge_desc
    );
    builder.add_computational_memlet(block, tasklet, "_out", B, {symbolic::symbol("i")}, edge_desc);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    analysis::DataDependencyAnalysis analysis(sdfg, true);
    analysis.run(analysis_manager);
    auto& dependencies = analysis.dependencies(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 0);
}

TEST(LoopDependencyAnalysisTest, Gather_1D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Int64);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("B", desc, true);
    builder.add_container("C", desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("b", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block_1 = builder.add_block(body);

    // Define indirection
    auto& A = builder.add_access(block_1, "A");
    auto& b = builder.add_access(block_1, "b");
    auto& tasklet1 = builder.add_tasklet(block_1, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block_1, A, tasklet1, "_in", {indvar}, edge_desc);
    builder.add_computational_memlet(block_1, tasklet1, "_out", b, {});

    auto& block_2 = builder.add_block(body);
    auto& B = builder.add_access(block_2, "B");
    auto& C = builder.add_access(block_2, "C");
    auto& tasklet = builder.add_tasklet(block_2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block_2, B, tasklet, "_in", {symbolic::symbol("b")}, edge_desc);
    builder.add_computational_memlet(block_2, tasklet, "_out", C, {symbolic::symbol("i")}, edge_desc);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    analysis::DataDependencyAnalysis analysis(sdfg, true);
    analysis.run(analysis_manager);
    auto& dependencies = analysis.dependencies(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 1);
    EXPECT_EQ(dependencies.at("b"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
}

TEST(LoopDependencyAnalysisTest, Scatter_1D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Int64);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("B", desc, true);
    builder.add_container("C", desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("b", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Define indirection
    auto& block_1 = builder.add_block(body);
    auto& A = builder.add_access(block_1, "A");
    auto& b = builder.add_access(block_1, "b");
    auto& tasklet1 = builder.add_tasklet(block_1, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block_1, A, tasklet1, "_in", {symbolic::symbol("i")}, edge_desc);
    builder.add_computational_memlet(block_1, tasklet1, "_out", b, {});

    auto& block_2 = builder.add_block(body);
    auto& B = builder.add_access(block_2, "B");
    auto& C = builder.add_access(block_2, "C");
    auto& tasklet = builder.add_tasklet(block_2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block_2, B, tasklet, "_in", {symbolic::symbol("i")}, edge_desc);
    builder.add_computational_memlet(block_2, tasklet, "_out", C, {symbolic::symbol("b")}, edge_desc);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    analysis::DataDependencyAnalysis analysis(sdfg, true);
    analysis.run(analysis_manager);
    auto& dependencies = analysis.dependencies(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 2);
    EXPECT_EQ(dependencies.at("b"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
    EXPECT_EQ(dependencies.at("C"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
}

TEST(LoopDependencyAnalysisTest, MapDeg2_1D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& A = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"0"});
    builder.add_computational_memlet(
        block, tasklet, "_out", A, {symbolic::mul(symbolic::symbol("i"), symbolic::symbol("i"))}, edge_desc
    );

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    analysis::DataDependencyAnalysis analysis(sdfg, true);
    analysis.run(analysis_manager);
    auto& dependencies = analysis.dependencies(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 0);
}

TEST(LoopDependencyAnalysisTest, Map_2D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_desc(base_desc, symbolic::symbol("M"));
    types::Pointer edge_desc(array_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    // Define loop 1
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Define loop 2
    auto bound_2 = symbolic::symbol("M");
    auto indvar_2 = symbolic::symbol("j");
    auto init_2 = symbolic::integer(0);
    auto condition_2 = symbolic::Lt(indvar_2, bound_2);
    auto update_2 = symbolic::add(indvar_2, symbolic::integer(1));

    auto& loop_2 = builder.add_for(body, indvar_2, condition_2, init_2, update_2);
    auto& body_2 = loop_2.root();

    // Add computation
    auto& block = builder.add_block(body_2);
    auto& a_in = builder.add_access(block, "A");
    auto& one_node = builder.add_constant(block, "1.0", base_desc);
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder
        .add_computational_memlet(block, a_in, tasklet, "_in1", {symbolic::symbol("i"), symbolic::symbol("j")}, edge_desc);
    builder.add_computational_memlet(block, one_node, tasklet, "_in2", {});
    builder
        .add_computational_memlet(block, tasklet, "_out", a_out, {symbolic::symbol("i"), symbolic::symbol("j")}, edge_desc);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    analysis::DataDependencyAnalysis analysis(sdfg, true);
    analysis.run(analysis_manager);

    // Check
    auto& dependencies = analysis.dependencies(loop);
    EXPECT_EQ(dependencies.size(), 1);
    EXPECT_EQ(dependencies.at("j"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);

    // Check loop 2
    auto& dependencies_2 = analysis.dependencies(loop_2);
    EXPECT_EQ(dependencies_2.size(), 0);
}

TEST(LoopDependencyAnalysisTest, PartialSumInner_2D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_desc(base_desc, symbolic::symbol("N"));
    types::Pointer edge_desc(array_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("B", array_desc, true);

    // Define loop
    auto bound1 = symbolic::symbol("N");
    auto indvar1 = symbolic::symbol("i");
    auto init1 = symbolic::integer(0);
    auto condition1 = symbolic::Lt(indvar1, bound1);
    auto update1 = symbolic::add(indvar1, symbolic::integer(1));

    auto& loop1 = builder.add_for(root, indvar1, condition1, init1, update1);
    auto& body1 = loop1.root();

    auto bound2 = symbolic::symbol("N");
    auto indvar2 = symbolic::symbol("j");
    auto init2 = symbolic::integer(0);
    auto condition2 = symbolic::Lt(indvar2, bound2);
    auto update3 = symbolic::add(indvar2, symbolic::integer(1));

    auto& loop2 = builder.add_for(body1, indvar2, condition2, init2, update3);
    auto& body2 = loop2.root();

    // Add computation
    auto& block = builder.add_block(body2);
    auto& A = builder.add_access(block, "A");
    auto& B1 = builder.add_access(block, "B");
    auto& B2 = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder
        .add_computational_memlet(block, A, tasklet, "_in1", {symbolic::symbol("i"), symbolic::symbol("j")}, edge_desc);
    builder.add_computational_memlet(block, B1, tasklet, "_in2", {symbolic::symbol("i")}, array_desc);
    builder.add_computational_memlet(block, tasklet, "_out", B2, {symbolic::symbol("i")}, array_desc);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    analysis::DataDependencyAnalysis analysis(sdfg, true);
    analysis.run(analysis_manager);
    auto& dependencies1 = analysis.dependencies(loop1);
    auto& dependencies2 = analysis.dependencies(loop2);

    // Check
    EXPECT_EQ(dependencies1.size(), 1);
    EXPECT_EQ(dependencies1.at("j"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);

    EXPECT_EQ(dependencies2.size(), 1);
    EXPECT_EQ(dependencies2.at("B"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_READ_WRITE);
}

TEST(LoopDependencyAnalysisTest, PartialSumOuter_2D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_desc(base_desc, symbolic::symbol("N"));
    types::Pointer edge_desc(array_desc);
    types::Pointer desc;

    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("A", desc, true);
    builder.add_container("B", array_desc, true);

    // Define loop
    auto bound1 = symbolic::symbol("N");
    auto indvar1 = symbolic::symbol("i");
    auto init1 = symbolic::integer(0);
    auto condition1 = symbolic::Lt(indvar1, bound1);
    auto update1 = symbolic::add(indvar1, symbolic::integer(1));

    auto& loop1 = builder.add_for(root, indvar1, condition1, init1, update1);
    auto& body1 = loop1.root();

    auto bound2 = symbolic::symbol("N");
    auto indvar2 = symbolic::symbol("j");
    auto init2 = symbolic::integer(0);
    auto condition2 = symbolic::Lt(indvar2, bound2);
    auto update3 = symbolic::add(indvar2, symbolic::integer(1));

    auto& loop2 = builder.add_for(body1, indvar2, condition2, init2, update3);
    auto& body2 = loop2.root();

    // Add computation
    auto& block = builder.add_block(body2);
    auto& A = builder.add_access(block, "A");
    auto& B1 = builder.add_access(block, "B");
    auto& B2 = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, A, tasklet, "_in1", {indvar1, indvar2}, edge_desc);
    builder.add_computational_memlet(block, B1, tasklet, "_in2", {indvar2}, array_desc);
    builder.add_computational_memlet(block, tasklet, "_out", B2, {indvar2}, array_desc);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    analysis::DataDependencyAnalysis analysis(sdfg, true);
    analysis.run(analysis_manager);
    auto& dependencies1 = analysis.dependencies(loop1);
    auto& dependencies2 = analysis.dependencies(loop2);

    // Check
    EXPECT_EQ(dependencies1.size(), 2);
    EXPECT_EQ(dependencies1.at("j"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
    EXPECT_EQ(dependencies1.at("B"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_READ_WRITE);

    EXPECT_EQ(dependencies2.size(), 0);
}

TEST(LoopDependencyAnalysisTest, PartialSum_1D_Triangle) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer ptr_desc(base_desc);

    types::Pointer desc;
    builder.add_container("A", desc, true);

    // Define loop
    auto indvar1 = symbolic::symbol("i");
    auto init1 = symbolic::integer(0);
    auto condition1 = symbolic::Lt(indvar1, symbolic::symbol("N"));
    auto update1 = symbolic::add(indvar1, symbolic::integer(1));

    auto& loop1 = builder.add_for(root, indvar1, condition1, init1, update1);
    auto& body1 = loop1.root();

    // init block
    auto& init_block = builder.add_block(body1);
    auto& A_init = builder.add_access(init_block, "A");
    auto& zero_node = builder.add_constant(init_block, "0.0", base_desc);
    auto& tasklet_init = builder.add_tasklet(init_block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(init_block, zero_node, tasklet_init, "_in", {});
    builder.add_computational_memlet(init_block, tasklet_init, "_out", A_init, {indvar1}, ptr_desc);

    auto indvar2 = symbolic::symbol("j");
    auto init2 = symbolic::zero();
    auto condition2 = symbolic::Lt(indvar2, indvar1);
    auto update2 = symbolic::add(indvar2, symbolic::integer(1));

    auto& loop2 = builder.add_for(body1, indvar2, condition2, init2, update2);
    auto& body2 = loop2.root();

    // Reduction block
    auto& block = builder.add_block(body2);
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, A_in, tasklet, "_in1", {indvar2}, ptr_desc);
    builder.add_computational_memlet(block, A_in, tasklet, "_in2", {indvar1}, ptr_desc);
    builder.add_computational_memlet(block, tasklet, "_out", A_out, {indvar1}, ptr_desc);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    analysis::DataDependencyAnalysis analysis(sdfg, true);
    analysis.run(analysis_manager);
    auto& dependencies1 = analysis.dependencies(loop1);
    auto& dependencies2 = analysis.dependencies(loop2);

    // Check
    EXPECT_EQ(dependencies1.size(), 2);
    EXPECT_EQ(dependencies1.at("A"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_READ_WRITE);
    EXPECT_EQ(dependencies1.at("j"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
    EXPECT_EQ(dependencies2.size(), 1);
    EXPECT_EQ(dependencies2.at("A"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_READ_WRITE);
}

TEST(LoopDependencyAnalysisTest, Transpose_2D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_desc(base_desc, symbolic::symbol("M"));
    types::Pointer edge_desc(array_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("B", desc, true);

    // Define loop
    auto bound1 = symbolic::symbol("N");
    auto indvar1 = symbolic::symbol("i");
    auto init1 = symbolic::integer(0);
    auto condition1 = symbolic::Lt(indvar1, bound1);
    auto update1 = symbolic::add(indvar1, symbolic::integer(1));

    auto& loop1 = builder.add_for(root, indvar1, condition1, init1, update1);
    auto& body1 = loop1.root();

    auto bound2 = symbolic::symbol("M");
    auto indvar2 = symbolic::symbol("j");
    auto init2 = symbolic::integer(0);
    auto condition2 = symbolic::Lt(indvar2, bound2);
    auto update3 = symbolic::add(indvar2, symbolic::integer(1));

    auto& loop2 = builder.add_for(body1, indvar2, condition2, init2, update3);
    auto& body2 = loop2.root();

    // Add computation
    auto& block = builder.add_block(body2);
    auto& A = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder
        .add_computational_memlet(block, A, tasklet, "_in", {symbolic::symbol("i"), symbolic::symbol("j")}, edge_desc);
    builder
        .add_computational_memlet(block, tasklet, "_out", B, {symbolic::symbol("j"), symbolic::symbol("i")}, edge_desc);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    analysis::DataDependencyAnalysis analysis(sdfg, true);
    analysis.run(analysis_manager);
    auto& dependencies1 = analysis.dependencies(loop1);
    auto& dependencies2 = analysis.dependencies(loop2);

    // Check
    EXPECT_EQ(dependencies2.size(), 0);
    EXPECT_EQ(dependencies1.size(), 1);
    EXPECT_EQ(dependencies1.at("j"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
}

TEST(LoopDependencyAnalysisTest, TransposeTriangle_2D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_desc(base_desc, symbolic::symbol("N"));
    types::Pointer edge_desc(array_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);

    // Define loop
    auto indvar1 = symbolic::symbol("i");
    auto init1 = symbolic::integer(0);
    auto condition1 = symbolic::Lt(indvar1, symbolic::symbol("N"));
    auto update1 = symbolic::add(indvar1, symbolic::integer(1));

    auto& loop1 = builder.add_for(root, indvar1, condition1, init1, update1);
    auto& body1 = loop1.root();

    auto indvar2 = symbolic::symbol("j");
    auto init2 = symbolic::add(indvar1, symbolic::integer(1));
    auto condition2 = symbolic::Lt(indvar2, symbolic::symbol("N"));
    auto update2 = symbolic::add(indvar2, symbolic::integer(1));

    auto& loop2 = builder.add_for(body1, indvar2, condition2, init2, update2);
    auto& body2 = loop2.root();

    // Add computation
    auto& block = builder.add_block(body2);
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder
        .add_computational_memlet(block, A_in, tasklet, "_in", {symbolic::symbol("i"), symbolic::symbol("j")}, edge_desc);
    builder
        .add_computational_memlet(block, tasklet, "_out", A_out, {symbolic::symbol("j"), symbolic::symbol("i")}, edge_desc);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    analysis::DataDependencyAnalysis analysis(sdfg, true);
    analysis.run(analysis_manager);
    auto& dependencies1 = analysis.dependencies(loop1);
    auto& dependencies2 = analysis.dependencies(loop2);

    // Check
    EXPECT_EQ(dependencies1.size(), 1);
    EXPECT_EQ(dependencies1.at("j"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
    EXPECT_EQ(dependencies2.size(), 0);
}

TEST(LoopDependencyAnalysisTest, TransposeTriangleWithDiagonal_2D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_desc(base_desc, symbolic::symbol("N"));
    types::Pointer edge_desc(array_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);

    // Define loop
    auto indvar1 = symbolic::symbol("i");
    auto init1 = symbolic::integer(0);
    auto condition1 = symbolic::Lt(indvar1, symbolic::symbol("N"));
    auto update1 = symbolic::add(indvar1, symbolic::integer(1));

    auto& loop1 = builder.add_for(root, indvar1, condition1, init1, update1);
    auto& body1 = loop1.root();

    auto indvar2 = symbolic::symbol("j");
    auto init2 = indvar1;
    auto condition2 = symbolic::Lt(indvar2, symbolic::symbol("N"));
    auto update2 = symbolic::add(indvar2, symbolic::integer(1));

    auto& loop2 = builder.add_for(body1, indvar2, condition2, init2, update2);
    auto& body2 = loop2.root();

    // Add computation
    auto& block = builder.add_block(body2);
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder
        .add_computational_memlet(block, A_in, tasklet, "_in", {symbolic::symbol("i"), symbolic::symbol("j")}, edge_desc);
    builder
        .add_computational_memlet(block, tasklet, "_out", A_out, {symbolic::symbol("j"), symbolic::symbol("i")}, edge_desc);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    analysis::DataDependencyAnalysis analysis(sdfg, true);
    analysis.run(analysis_manager);
    auto& dependencies1 = analysis.dependencies(loop1);
    auto& dependencies2 = analysis.dependencies(loop2);

    // Check
    EXPECT_EQ(dependencies1.size(), 1);
    EXPECT_EQ(dependencies1.at("j"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
    EXPECT_EQ(dependencies2.size(), 0);
}

TEST(LoopDependencyAnalysisTest, TransposeSquare_2D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_desc(base_desc, symbolic::symbol("N"));
    types::Pointer edge_desc(array_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);

    // Define loop
    auto bound1 = symbolic::symbol("N");
    auto indvar1 = symbolic::symbol("i");
    auto init1 = symbolic::integer(0);
    auto condition1 = symbolic::Lt(indvar1, bound1);
    auto update1 = symbolic::add(indvar1, symbolic::integer(1));

    auto& loop1 = builder.add_for(root, indvar1, condition1, init1, update1);
    auto& body1 = loop1.root();

    auto bound2 = symbolic::symbol("N");
    auto indvar2 = symbolic::symbol("j");
    auto init2 = symbolic::integer(0);
    auto condition2 = symbolic::Lt(indvar2, bound2);
    auto update3 = symbolic::add(indvar2, symbolic::integer(1));

    auto& loop2 = builder.add_for(body1, indvar2, condition2, init2, update3);
    auto& body2 = loop2.root();

    // Add computation
    auto& block = builder.add_block(body2);
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, A_in, tasklet, "_in", {indvar1, indvar2}, edge_desc);
    builder.add_computational_memlet(block, tasklet, "_out", A_out, {indvar2, indvar1}, edge_desc);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    analysis::DataDependencyAnalysis analysis(sdfg, true);
    analysis.run(analysis_manager);
    auto& dependencies1 = analysis.dependencies(loop1);
    auto& dependencies2 = analysis.dependencies(loop2);

    // Check
    EXPECT_EQ(dependencies2.size(), 0);
    EXPECT_EQ(dependencies1.size(), 2);
    EXPECT_EQ(dependencies1.at("A"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_READ_WRITE);
    EXPECT_EQ(dependencies1.at("j"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
}

TEST(LoopDependencyAnalysisTest, ReductionWithLocalStorage) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer ptr_desc(base_desc);
    types::Array array_desc(base_desc, symbolic::integer(2));

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("B", opaque_desc, true);
    builder.add_container("C", opaque_desc, true);
    builder.add_container("local", array_desc);

    // Define loop
    auto indvar1 = symbolic::symbol("i");
    auto init1 = symbolic::integer(0);
    auto condition1 = symbolic::Lt(indvar1, symbolic::symbol("N"));
    auto update1 = symbolic::add(indvar1, symbolic::integer(1));

    auto& loop1 = builder.add_for(root, indvar1, condition1, init1, update1);
    auto& body1 = loop1.root();

    // local[0] = 0.0 block
    {
        auto& init_block = builder.add_block(body1);
        auto& local_init_0 = builder.add_access(init_block, "local");
        auto& zero_node = builder.add_constant(init_block, "0.0", base_desc);
        auto& tasklet_init = builder.add_tasklet(init_block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(init_block, zero_node, tasklet_init, "_in", {});
        builder.add_computational_memlet(init_block, tasklet_init, "_out", local_init_0, {symbolic::zero()}, array_desc);
    }

    // local[1] = 0.0 block
    {
        auto& init_block = builder.add_block(body1);
        auto& local_init_0 = builder.add_access(init_block, "local");
        auto& zero_node = builder.add_constant(init_block, "0.0", base_desc);
        auto& tasklet_init = builder.add_tasklet(init_block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(init_block, zero_node, tasklet_init, "_in", {});
        builder.add_computational_memlet(init_block, tasklet_init, "_out", local_init_0, {symbolic::one()}, array_desc);
    }

    auto indvar2 = symbolic::symbol("j");
    auto init2 = symbolic::zero();
    auto condition2 = symbolic::Lt(indvar2, indvar1);
    auto update2 = symbolic::add(indvar2, symbolic::integer(1));

    auto& loop2 = builder.add_for(body1, indvar2, condition2, init2, update2);
    auto& body2 = loop2.root();

    // Reduction: local[0] += A[j] block
    {
        auto& block = builder.add_block(body2);
        auto& A_in = builder.add_access(block, "A");
        auto& local_in = builder.add_access(block, "local");
        auto& local_out = builder.add_access(block, "local");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
        builder.add_computational_memlet(block, A_in, tasklet, "_in1", {indvar2}, ptr_desc);
        builder.add_computational_memlet(block, local_in, tasklet, "_in2", {symbolic::zero()}, array_desc);
        builder.add_computational_memlet(block, tasklet, "_out", local_out, {symbolic::zero()}, array_desc);
    }

    // Reduction: local[1] *= A[j] block
    {
        auto& block = builder.add_block(body2);
        auto& A_in = builder.add_access(block, "A");
        auto& local_in = builder.add_access(block, "local");
        auto& local_out = builder.add_access(block, "local");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
        builder.add_computational_memlet(block, A_in, tasklet, "_in1", {indvar2}, ptr_desc);
        builder.add_computational_memlet(block, local_in, tasklet, "_in2", {symbolic::one()}, array_desc);
        builder.add_computational_memlet(block, tasklet, "_out", local_out, {symbolic::one()}, array_desc);
    }

    // Writeback block: B[i] = local[0]; C[i] = local[1]
    {
        auto& block = builder.add_block(body1);
        auto& local_in_0 = builder.add_access(block, "local");
        auto& local_in_1 = builder.add_access(block, "local");
        auto& B_out = builder.add_access(block, "B");
        auto& C_out = builder.add_access(block, "C");
        auto& tasklet_0 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, local_in_0, tasklet_0, "_in", {symbolic::zero()}, array_desc);
        builder.add_computational_memlet(block, tasklet_0, "_out", B_out, {indvar1}, ptr_desc);

        auto& tasklet_1 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, local_in_1, tasklet_1, "_in", {symbolic::one()}, array_desc);
        builder.add_computational_memlet(block, tasklet_1, "_out", C_out, {indvar1}, ptr_desc);
    }

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    analysis::DataDependencyAnalysis analysis(sdfg, true);
    analysis.run(analysis_manager);
    auto& dependencies1 = analysis.dependencies(loop1);
    auto& dependencies2 = analysis.dependencies(loop2);

    // Check
    EXPECT_EQ(dependencies1.size(), 2);
    EXPECT_EQ(dependencies1.at("local"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
    EXPECT_EQ(dependencies1.at("j"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
    EXPECT_EQ(dependencies2.size(), 1);
    EXPECT_EQ(dependencies2.at("local"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_READ_WRITE);
}
