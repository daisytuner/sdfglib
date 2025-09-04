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
    analysis::DataDependencyAnalysis analysis(builder.subject());
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;

    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    analysis.visit_block(users, assumptions_analysis, block, undefined, open_definitions, closed_definitions);

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
    analysis::DataDependencyAnalysis analysis(builder.subject());
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;

    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    analysis.visit_block(users, assumptions_analysis, block, undefined, open_definitions, closed_definitions);

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
    analysis::DataDependencyAnalysis analysis(builder.subject());
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;

    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    analysis.visit_block(users, assumptions_analysis, block, undefined, open_definitions, closed_definitions);

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
    analysis::DataDependencyAnalysis analysis(builder.subject());
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;

    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    analysis.visit_block(users, assumptions_analysis, block, undefined, open_definitions, closed_definitions);

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
    analysis::DataDependencyAnalysis analysis(builder.subject());
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;

    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    analysis.visit_block(users, assumptions_analysis, block, undefined, open_definitions, closed_definitions);

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
    analysis::DataDependencyAnalysis analysis(builder.subject());
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;

    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    analysis.visit_block(users, assumptions_analysis, block, undefined, open_definitions, closed_definitions);

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
    analysis::DataDependencyAnalysis analysis(builder.subject());
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;

    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    analysis.visit_block(users, assumptions_analysis, block, undefined, open_definitions, closed_definitions);

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
    analysis::DataDependencyAnalysis analysis(builder.subject());
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;

    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    analysis.visit_block(users, assumptions_analysis, block, undefined, open_definitions, closed_definitions);

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
    analysis::DataDependencyAnalysis analysis(builder.subject());
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;

    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    analysis.visit_sequence(users, assumptions_analysis, root, undefined, open_definitions, closed_definitions);

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
    analysis::DataDependencyAnalysis analysis(builder.subject());
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;

    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    analysis.visit_sequence(users, assumptions_analysis, root, undefined, open_definitions, closed_definitions);

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
    analysis::DataDependencyAnalysis analysis(builder.subject());
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;

    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    analysis.visit_sequence(users, assumptions_analysis, root, undefined, open_definitions, closed_definitions);

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
    auto& tasklet = builder.add_tasklet(block1, data_flow::TaskletCode::assign, "_out", {"0"});
    builder.add_computational_memlet(block1, tasklet, "_out", output_node, {symbolic::integer(0)});

    auto& block2 = builder.add_block(root);
    auto& output_node2 = builder.add_access(block2, "A");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"1"});
    builder.add_computational_memlet(block2, tasklet2, "_out", output_node2, {symbolic::integer(0)});

    // Run analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    analysis::DataDependencyAnalysis analysis(builder.subject());
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;

    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    analysis.visit_sequence(users, assumptions_analysis, root, undefined, open_definitions, closed_definitions);

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
    analysis::DataDependencyAnalysis analysis(builder.subject());
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;

    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    analysis.visit_sequence(users, assumptions_analysis, root, undefined, open_definitions, closed_definitions);

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
    analysis::DataDependencyAnalysis analysis(builder.subject());
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;

    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    analysis.visit_sequence(users, assumptions_analysis, root, undefined, open_definitions, closed_definitions);

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
    analysis::DataDependencyAnalysis analysis(builder.subject());
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;

    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    analysis.visit_sequence(users, assumptions_analysis, root, undefined, open_definitions, closed_definitions);

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
    analysis::DataDependencyAnalysis analysis(builder.subject());
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;

    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    analysis.visit_sequence(users, assumptions_analysis, root, undefined, open_definitions, closed_definitions);

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
    analysis::DataDependencyAnalysis analysis(builder.subject());
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;

    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    analysis.visit_sequence(users, assumptions_analysis, root, undefined, open_definitions, closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 0);
    EXPECT_EQ(open_definitions.size(), 3);
    EXPECT_EQ(closed_definitions.size(), 0);

    auto& transition0 = root.at(0).second;
    auto& transition1 = branch1.at(0).second;
    auto& transition3 = root.at(2).second;

    auto write_A_0 = users.get_user("A", &transition0, analysis::Use::WRITE);
    auto write_A_1 = users.get_user("A", &transition1, analysis::Use::WRITE);
    auto write_B = users.get_user("B", &transition3, analysis::Use::WRITE);
    auto read_A = users.get_user("A", &transition3, analysis::Use::READ);

    auto& definition_A_0 = open_definitions.at(write_A_0);
    EXPECT_EQ(definition_A_0.size(), 2);
    EXPECT_NE(definition_A_0.find(read_A), definition_A_0.end());
    EXPECT_NE(definition_A_0.find(write_A_1), definition_A_0.end());

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
    auto& analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;

    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    analysis.visit_sequence(users, assumptions_analysis, root, undefined, open_definitions, closed_definitions);

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
    analysis::DataDependencyAnalysis analysis(builder.subject());
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;

    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    analysis.visit_sequence(users, assumptions_analysis, root, undefined, open_definitions, closed_definitions);

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
    analysis::DataDependencyAnalysis analysis(builder.subject());
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;

    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    analysis.visit_sequence(users, assumptions_analysis, root, undefined, open_definitions, closed_definitions);

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
    analysis::DataDependencyAnalysis analysis(builder.subject());
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;

    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    analysis.visit_sequence(users, assumptions_analysis, root, undefined, open_definitions, closed_definitions);

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
    analysis::DataDependencyAnalysis analysis(builder.subject());
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;

    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    analysis.visit_sequence(users, assumptions_analysis, root, undefined, open_definitions, closed_definitions);

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
    analysis::DataDependencyAnalysis analysis(builder.subject());
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;

    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    analysis.visit_sequence(users, assumptions_analysis, root, undefined, open_definitions, closed_definitions);

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
    analysis::DataDependencyAnalysis analysis(builder.subject());
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;

    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    analysis.visit_sequence(users, assumptions_analysis, root, undefined, open_definitions, closed_definitions);

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
    analysis::DataDependencyAnalysis analysis(builder.subject());
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;

    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    analysis.visit_sequence(users, assumptions_analysis, root, undefined, open_definitions, closed_definitions);

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

/*
TEST(DataDependencyAnalysisTest, visit_map) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    builder.add_container("B", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));

    auto& root = builder.subject().root();

    auto& map = builder.add_map(root, symbolic::symbol("i"), symbolic::integer(10),
                                structured_control_flow::ScheduleType_Sequential);

    auto& block = builder.add_block(map.root());
    auto& input_node = builder.add_access(block, "i");
    auto& output_node = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                        "_out",
                                        {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {},
types::Scalar(types::PrimitiveType::Int32)); builder.add_computational_memlet(block, input_node, tasklet, "_in", {},
types::Scalar(types::PrimitiveType::Int32));

    auto sdfg = builder.move();

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;

    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    symbolic::Assumptions assumptions = assumptions_analysis.get(map, true);
    analysis.visit_map(users, assumptions_analysis, assumptions, map, undefined, open_definitions,
                       closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 0);
    EXPECT_EQ(open_definitions.size(), 2);
    EXPECT_EQ(closed_definitions.size(), 0);

    bool foundB = false;
    bool foundi = false;

    for (auto entry : open_definitions) {
        if (entry.first->container() == "B") {
            foundB = true;
            EXPECT_EQ(entry.first->use(), analysis::Use::WRITE);
            EXPECT_EQ(entry.first->container(), "B");
            EXPECT_EQ(entry.first->element(), &output_node);
            EXPECT_EQ(entry.second.size(), 0);
        } else if (entry.first->container() == "i") {
            foundi = true;
            EXPECT_EQ(entry.first->use(), analysis::Use::WRITE);
            EXPECT_EQ(entry.first->container(), "i");
            EXPECT_EQ(entry.first->element(), &map);
            EXPECT_EQ(entry.second.size(), 1);
        }
    }

    EXPECT_TRUE(foundB);
    EXPECT_TRUE(foundi);
}

TEST(DataDependencyAnalysisTest, visit_while) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    builder.add_container("A", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("B", types::Scalar(types::PrimitiveType::Int32));

    auto& root = builder.subject().root();

    auto& while_loop = builder.add_while(root);

    auto& block = builder.add_block(while_loop.root());
    auto& input_node = builder.add_access(block, "A");
    auto& output_node = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                        "_out",
                                        {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {},
types::Scalar(types::PrimitiveType::Int32)); builder.add_computational_memlet(block, input_node, tasklet, "_in", {},
types::Scalar(types::PrimitiveType::Int32));

    auto& output_node2 = builder.add_access(block, "A");
    auto& tasklet2 = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                         "_out",
                                         {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_computational_memlet(block, tasklet2, "_out", output_node2, {},
types::Scalar(types::PrimitiveType::Int32)); builder.add_computational_memlet(block, output_node, tasklet2, "_in", {},
types::Scalar(types::PrimitiveType::Int32));

    auto sdfg = builder.move();

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;

    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    symbolic::Assumptions assumptions = assumptions_analysis.get(while_loop, true);
    analysis.visit_while(users, assumptions_analysis, assumptions, while_loop, undefined,
                         open_definitions, closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 1);
    auto& oread = *undefined.begin();
    EXPECT_EQ(oread->use(), analysis::Use::READ);
    EXPECT_EQ(oread->container(), "A");
    EXPECT_EQ(oread->element(), &input_node);

    EXPECT_EQ(open_definitions.size(), 2);
    EXPECT_EQ(closed_definitions.size(), 0);

    bool foundA = false;
    bool foundB = false;

    for (auto entry : open_definitions) {
        if (entry.first->container() == "A") {
            foundA = true;
            EXPECT_EQ(entry.first->use(), analysis::Use::WRITE);
            EXPECT_EQ(entry.first->container(), "A");
            EXPECT_EQ(entry.first->element(), &output_node2);
            EXPECT_EQ(entry.second.size(), 1);
            EXPECT_EQ((*entry.second.begin())->element(), &input_node);
        } else if (entry.first->container() == "B") {
            foundB = true;
            EXPECT_EQ(entry.first->use(), analysis::Use::WRITE);
            EXPECT_EQ(entry.first->container(), "B");
            EXPECT_EQ(entry.first->element(), &output_node);
            EXPECT_EQ(entry.second.size(), 1);
            EXPECT_EQ((*entry.second.begin())->element(), &output_node);
        }
    }
}

TEST(DataDependencyAnalysisTest, visit_sequence_for_loop) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    builder.add_container("A", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("B", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("C", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& input_node = builder.add_access(block, "A");
    auto& output_node = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                        "_out",
                                        {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {},
types::Scalar(types::PrimitiveType::Int32)); builder.add_computational_memlet(block, input_node, tasklet, "_in", {},
types::Scalar(types::PrimitiveType::Int32));

    auto& for_loop = builder.add_for(
        root, symbolic::symbol("i"), symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0), symbolic::add(symbolic::symbol("i"), symbolic::integer(1)));

    auto& block2 = builder.add_block(for_loop.root());
    auto& input_node2 = builder.add_access(block2, "i");
    auto& output_node2 = builder.add_access(block2, "A");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign,
                                         "_out",
                                         {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_computational_memlet(block2, tasklet2, "_out", output_node2, {},
types::Scalar(types::PrimitiveType::Int32)); builder.add_computational_memlet(block2, input_node2, tasklet2, "_in", {},
types::Scalar(types::PrimitiveType::Int32));

    auto sdfg = builder.move();

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;

    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    symbolic::Assumptions assumptions =
        assumptions_analysis.get(builder_opt.subject().root(), true);
    analysis.visit_sequence(users, assumptions_analysis, assumptions, builder_opt.subject().root(),
                            undefined, open_definitions, closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 1);
    EXPECT_EQ(open_definitions.size(), 4);
    EXPECT_EQ(closed_definitions.size(), 0);

    auto read = *undefined.begin();

    EXPECT_EQ(read->use(), analysis::Use::READ);
    EXPECT_EQ(read->container(), "A");
    EXPECT_EQ(read->element(), &input_node);

    bool foundB = false;
    bool foundA = false;
    int both_i = 0;

    for (auto entry : open_definitions) {
        if (entry.first->container() == "B") {
            foundB = true;
            EXPECT_EQ(entry.first->use(), analysis::Use::WRITE);
            EXPECT_EQ(entry.first->container(), "B");
            EXPECT_EQ(entry.first->element(), &output_node);
            EXPECT_EQ(entry.second.size(), 0);
        } else if (entry.first->container() == "A") {
            foundA = true;
            EXPECT_EQ(entry.first->use(), analysis::Use::WRITE);
            EXPECT_EQ(entry.first->container(), "A");
            EXPECT_EQ(entry.first->element(), &output_node2);
            EXPECT_EQ(entry.second.size(), 0);
        } else if (entry.first->container() == "i") {
            both_i++;
            EXPECT_EQ(entry.first->use(), analysis::Use::WRITE);
            EXPECT_EQ(entry.first->container(), "i");
            EXPECT_EQ(entry.first->element(), &for_loop);
            EXPECT_EQ(entry.second.size(), 3);
        }
    }

    EXPECT_TRUE(foundB && foundA);
    EXPECT_EQ(both_i, 2);
}

TEST(DataDependencyAnalysisTest, visit_sequence_while_loop) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    builder.add_container("A", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("B", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("C", types::Scalar(types::PrimitiveType::Int32));

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& input_node = builder.add_access(block, "A");
    auto& output_node = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                        "_out",
                                        {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {},
types::Scalar(types::PrimitiveType::Int32)); builder.add_computational_memlet(block, input_node, tasklet, "_in", {},
types::Scalar(types::PrimitiveType::Int32));

    auto& while_loop = builder.add_while(root);

    auto& block2 = builder.add_block(while_loop.root());
    auto& input_node2 = builder.add_access(block2, "B");
    auto& output_node2 = builder.add_access(block2, "A");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign,
                                         "_out",
                                         {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_computational_memlet(block2, tasklet2, "_out", output_node2, {},
types::Scalar(types::PrimitiveType::Int32)); builder.add_computational_memlet(block2, input_node2, tasklet2, "_in", {},
types::Scalar(types::PrimitiveType::Int32));

    auto sdfg = builder.move();

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;

    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    symbolic::Assumptions assumptions =
        assumptions_analysis.get(builder_opt.subject().root(), true);
    analysis.visit_sequence(users, assumptions_analysis, assumptions, builder_opt.subject().root(),
                            undefined, open_definitions, closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 1);
    EXPECT_EQ(open_definitions.size(), 2);
    EXPECT_EQ(closed_definitions.size(), 0);

    auto read = *undefined.begin();

    EXPECT_EQ(read->use(), analysis::Use::READ);
    EXPECT_EQ(read->container(), "A");
    EXPECT_EQ(read->element(), &input_node);

    bool foundB = false;
    bool foundA = false;

    for (auto entry : open_definitions) {
        if (entry.first->container() == "B") {
            foundB = true;
            EXPECT_EQ(entry.first->use(), analysis::Use::WRITE);
            EXPECT_EQ(entry.first->container(), "B");
            EXPECT_EQ(entry.first->element(), &output_node);
            EXPECT_EQ(entry.second.size(), 1);
            EXPECT_EQ((*entry.second.begin())->container(), "B");
        } else if (entry.first->container() == "A") {
            foundA = true;
            EXPECT_EQ(entry.first->use(), analysis::Use::WRITE);
            EXPECT_EQ(entry.first->container(), "A");
            EXPECT_EQ(entry.first->element(), &output_node2);
            EXPECT_EQ(entry.second.size(), 0);
        }
    }

    EXPECT_TRUE(foundB && foundA);
}

TEST(DataDependencyAnalysisTest, visit_sdfg) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    builder.add_container("A", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("B", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("C", types::Scalar(types::PrimitiveType::Int32));

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& input_node = builder.add_access(block, "A");
    auto& output_node = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                        "_out",
                                        {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {},
types::Scalar(types::PrimitiveType::Int32)); builder.add_computational_memlet(block, input_node, tasklet, "_in", {},
types::Scalar(types::PrimitiveType::Int32));

    auto& input_node3 = output_node;
    auto& output_node3 = builder.add_access(block, "C");
    auto& tasklet3 = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                         "_out",
                                         {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_computational_memlet(block, tasklet3, "_out", output_node3, {},
types::Scalar(types::PrimitiveType::Int32)); builder.add_computational_memlet(block, input_node3, tasklet3, "_in", {},
types::Scalar(types::PrimitiveType::Int32));

    auto& input_node2 = output_node3;
    auto& output_node2 = builder.add_access(block, "B");
    auto& tasklet2 = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                         "_out",
                                         {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_computational_memlet(block, tasklet2, "_out", output_node2, {},
types::Scalar(types::PrimitiveType::Int32)); builder.add_computational_memlet(block, input_node2, tasklet2, "_in", {},
types::Scalar(types::PrimitiveType::Int32));

    auto sdfg = builder.move();

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();
    auto& users = analysis_manager.get<analysis::Users>();

    // Check result
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> reads_after_writes_A =
        analysis.definitions("A");
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> reads_after_writes_B =
        analysis.definitions("B");
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> reads_after_writes_C =
        analysis.definitions("C");

    EXPECT_EQ(reads_after_writes_A.size(), 0);
    EXPECT_EQ(reads_after_writes_B.size(), 2);
    EXPECT_EQ(reads_after_writes_C.size(), 1);

    bool foundB_first = false;
    bool foundB_second = false;
    bool foundC = false;

    for (auto& entry : reads_after_writes_B) {
        if (entry.first->element() == &output_node) {
            foundB_first = true;
            EXPECT_EQ(entry.first->use(), analysis::Use::WRITE);
            EXPECT_EQ(entry.first->container(), "B");
            EXPECT_EQ(entry.second.size(), 1);
            EXPECT_EQ((*entry.second.begin())->container(), "B");
            EXPECT_EQ((*entry.second.begin())->element(), &input_node3);
        } else if (entry.first->element() == &output_node2) {
            foundB_second = true;
            EXPECT_EQ(entry.first->use(), analysis::Use::WRITE);
            EXPECT_EQ(entry.first->container(), "B");
            EXPECT_EQ(entry.second.size(), 0);
        }
    }

    for (auto& entry : reads_after_writes_C) {
        foundC = true;
        EXPECT_EQ(entry.first->use(), analysis::Use::WRITE);
        EXPECT_EQ(entry.first->container(), "C");
        EXPECT_EQ(entry.second.size(), 1);
        EXPECT_EQ((*entry.second.begin())->container(), "C");
        EXPECT_EQ((*entry.second.begin())->element(), &input_node2);
    }

    EXPECT_TRUE(foundB_first && foundB_second && foundC);
}

TEST(DataDependencyAnalysisTest, propagate_open_read_out_of_while) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    builder.add_container("_0", types::Pointer(types::Scalar(types::PrimitiveType::Double)), true);
    builder.add_container("_1", types::Pointer(types::Scalar(types::PrimitiveType::Double)));
    builder.add_container("_7", types::Pointer(types::Scalar(types::PrimitiveType::Double)));
    builder.add_container("_16", types::Pointer(types::Scalar(types::PrimitiveType::Double)));

    builder.add_container("_4", types::Scalar(types::PrimitiveType::Int32));
    auto sym = symbolic::symbol("_4");

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root, {{sym, symbolic::integer(0)}});
    auto& outer_if_else = builder.add_if_else(root);
    auto& outer_case1 = builder.add_case(outer_if_else, symbolic::__false__());
    auto& outer_case2 = builder.add_case(outer_if_else, symbolic::__true__());

    auto& outer_loop = builder.add_while(outer_case2);
    auto& outer_body = outer_loop.root();
    auto& outer_block = builder.add_block(outer_body);

    auto& outer_input_node = builder.add_access(outer_block, "_1");
    auto& outer_output_node = builder.add_access(outer_block, "_16");
    auto& memlet =
        builder.add_computational_memlet(outer_block, outer_input_node, outer_output_node, "refs", {sym},
types::Pointer(types::Scalar(types::PrimitiveType::Double)));

    auto& increment_block =
        builder.add_block(outer_body, {{sym, symbolic::add(sym, symbolic::integer(1))}});
    auto& outer_cont_break = builder.add_if_else(outer_body);
    auto& outer_cont_case =
        builder.add_case(outer_cont_break, symbolic::Lt(sym, symbolic::integer(10)));
    auto& outer_cont = builder.add_continue(outer_cont_case);
    auto& outer_break_case =
        builder.add_case(outer_cont_break, symbolic::Ge(sym, symbolic::integer(10)));
    auto& outer_break = builder.add_break(outer_break_case);

    auto sdfg = builder.move();

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();
    auto& users = analysis_manager.get<analysis::Users>();

    // Check result
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> reads_after_writes_4 =
        analysis.definitions("_4");

    EXPECT_EQ(reads_after_writes_4.size(), 2);
    for (auto& entry : reads_after_writes_4) {
        EXPECT_EQ(entry.first->use(), analysis::Use::WRITE);
        EXPECT_EQ(entry.first->container(), "_4");

        if (entry.first->element() == &root.at(0).second) {
            EXPECT_EQ(entry.second.size(), 2);
            for (auto& entry2 : entry.second) {
                if (entry2->element() == &memlet) {
                    EXPECT_EQ(entry2->use(), analysis::Use::READ);
                    EXPECT_EQ(entry2->container(), "_4");
                } else if (entry2->element() == &outer_body.at(1).second) {
                    EXPECT_EQ(entry2->use(), analysis::Use::READ);
                    EXPECT_EQ(entry2->container(), "_4");
                } else {
                    EXPECT_FALSE(true);
                }
            }
        } else if (entry.first->element() == &outer_body.at(1).second) {
            EXPECT_EQ(entry.second.size(), 3);
            for (auto& entry2 : entry.second) {
                if (entry2->element() == &memlet) {
                    EXPECT_EQ(entry2->use(), analysis::Use::READ);
                    EXPECT_EQ(entry2->container(), "_4");
                } else if (entry2->element() == &outer_cont_break) {
                    EXPECT_EQ(entry2->use(), analysis::Use::READ);
                    EXPECT_EQ(entry2->container(), "_4");
                } else if (entry2->element() == &outer_body.at(1).second) {
                    EXPECT_EQ(entry2->use(), analysis::Use::READ);
                    EXPECT_EQ(entry2->container(), "_4");
                } else {
                    EXPECT_FALSE(true);
                }
            }
        } else {
            EXPECT_FALSE(true);
        }
    }
}
*/

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
    auto& analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();
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
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, a1, tasklet, "_in1", {indvar}, edge_desc);
    builder.add_computational_memlet(block, b_in, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", b_out, {});

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();
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
    auto& analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();
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
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::add, "_out", {"_in1", "_in2"});
    builder
        .add_computational_memlet(block, A1, tasklet, "_in1", {symbolic::sub(indvar, symbolic::integer(1))}, edge_desc);
    builder.add_computational_memlet(block, A2, tasklet, "_in2", {indvar}, edge_desc);
    builder.add_computational_memlet(block, tasklet, "_out", A3, {indvar}, edge_desc);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();
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
    auto& analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();
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
    auto& analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();
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
    auto& analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();
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
    auto& analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();
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
    auto& analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();
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
    auto& i = builder.add_access(block, "i");
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in1", {symbolic::symbol("i")}, edge_desc);
    builder.add_computational_memlet(block, i, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", a_out, {symbolic::symbol("i")}, edge_desc);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();
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
    auto& i = builder.add_access(block, "i");
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in1", {symbolic::zero()}, edge_desc);
    builder.add_computational_memlet(block, i, tasklet, "_in2", {});
    builder.add_computational_memlet(
        block, tasklet, "_out", a_out, {symbolic::add(symbolic::symbol("i"), symbolic::one())}, edge_desc
    );

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();
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
    auto& analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();
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
    auto& analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();
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
    auto& i = builder.add_access(block, "i");
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in1", {symbolic::symbol("i")}, edge_desc);
    builder.add_computational_memlet(block, i, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", a_out, {symbolic::symbol("i")}, edge_desc);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();
    auto& dependencies1 = analysis.dependencies(loop_outer);
    auto& dependencies2 = analysis.dependencies(loop_inner);

    // Check
    EXPECT_EQ(dependencies1.size(), 1);
    EXPECT_EQ(dependencies1.at("i"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
    EXPECT_EQ(dependencies2.size(), 0);
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

    auto& assums_m = sdfg.assumption(symbolic::symbol("m"));
    assums_m.lower_bound(symbolic::integer(1));
    assums_m.constant(true);
    auto& assums_b = sdfg.assumption(symbolic::symbol("b"));
    assums_b.lower_bound(symbolic::integer(1));
    assums_b.constant(true);

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
    auto& analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();
    auto& dependencies = analysis.dependencies(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 0);
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
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fma, "_out", {"_in1", "_in2", "_in3"});
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
    auto& analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();
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
    auto& analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();
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
    auto& analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();
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
    auto& analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();
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
    auto& i = builder.add_access(block, "i");
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::add, "_out", {"_in1", "_in2"});
    builder
        .add_computational_memlet(block, a_in, tasklet, "_in1", {symbolic::symbol("i"), symbolic::symbol("j")}, edge_desc);
    builder.add_computational_memlet(block, i, tasklet, "_in2", {});
    builder
        .add_computational_memlet(block, tasklet, "_out", a_out, {symbolic::symbol("i"), symbolic::symbol("j")}, edge_desc);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();

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
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::add, "_out", {"_in1", "_in2"});
    builder
        .add_computational_memlet(block, A, tasklet, "_in1", {symbolic::symbol("i"), symbolic::symbol("j")}, edge_desc);
    builder.add_computational_memlet(block, B1, tasklet, "_in2", {symbolic::symbol("i")}, array_desc);
    builder.add_computational_memlet(block, tasklet, "_out", B2, {symbolic::symbol("i")}, array_desc);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();
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
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, A, tasklet, "_in1", {indvar1, indvar2}, edge_desc);
    builder.add_computational_memlet(block, B1, tasklet, "_in2", {indvar2}, array_desc);
    builder.add_computational_memlet(block, tasklet, "_out", B2, {indvar2}, array_desc);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();
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
    auto& tasklet_init = builder.add_tasklet(init_block, data_flow::TaskletCode::assign, "_out", {"0.0"});
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
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, A_in, tasklet, "_in1", {indvar2}, ptr_desc);
    builder.add_computational_memlet(block, A_in, tasklet, "_in2", {indvar1}, ptr_desc);
    builder.add_computational_memlet(block, tasklet, "_out", A_out, {indvar1}, ptr_desc);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();
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
    auto& analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();
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
    auto& analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();
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
    auto& analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();
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
    auto& analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();
    auto& dependencies1 = analysis.dependencies(loop1);
    auto& dependencies2 = analysis.dependencies(loop2);

    // Check
    EXPECT_EQ(dependencies2.size(), 0);
    EXPECT_EQ(dependencies1.size(), 2);
    EXPECT_EQ(dependencies1.at("A"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_READ_WRITE);
    EXPECT_EQ(dependencies1.at("j"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
}

TEST(LoopDependencyAnalysisTest, LUDecomposition_Blocked) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::Int32);
    builder.add_container("_0", sym_desc, true);

    types::Scalar sym_desc2(types::PrimitiveType::Int64);
    builder.add_container("_11", sym_desc2);
    builder.add_container("_19", sym_desc2);
    builder.add_container("_244", sym_desc2);
    builder.add_container("_303", sym_desc2);
    builder.add_container("_260", sym_desc2);
    builder.add_container("_267", sym_desc2);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("_1", desc, true);
    builder.add_container("_330", base_desc);
    builder.add_container("_258", base_desc);
    builder.add_container("_298", base_desc);

    // Loop _11
    auto bound_11 = symbolic::sub(symbolic::symbol("_0"), symbolic::integer(16));
    auto indvar_11 = symbolic::symbol("_11");
    auto init_11 = symbolic::integer(0);
    auto condition_11 = symbolic::Lt(indvar_11, bound_11);
    auto update_11 = symbolic::add(indvar_11, symbolic::integer(16));

    auto& loop_11 = builder.add_for(root, indvar_11, condition_11, init_11, update_11);
    auto& body_11 = loop_11.root();

    // Loop _19
    auto bound_19 = symbolic::integer(16);
    auto indvar_19 = symbolic::symbol("_19");
    auto init_19 = symbolic::integer(0);
    auto condition_19 = symbolic::Lt(indvar_19, bound_19);
    auto update_19 = symbolic::add(indvar_19, symbolic::integer(1));

    auto& loop_19 = builder.add_for(body_11, indvar_19, condition_19, init_19, update_19);
    auto& body_19 = loop_19.root();

    // Loop _244
    auto bound_244 = symbolic::integer(16);
    auto indvar_244 = symbolic::symbol("_244");
    auto init_244 = indvar_19;
    auto condition_244 = symbolic::Lt(indvar_244, bound_244);
    auto update_244 = symbolic::add(indvar_244, symbolic::integer(1));

    auto& loop_244 = builder.add_for(body_19, indvar_244, condition_244, init_244, update_244);
    auto& body_244 = loop_244.root();

    // Loop _303
    auto bound_303 = indvar_19;
    auto indvar_303 = symbolic::symbol("_303");
    auto init_303 = symbolic::integer(0);
    auto condition_303 = symbolic::Lt(indvar_303, bound_303);
    auto update_303 = symbolic::add(indvar_303, symbolic::integer(1));

    auto& loop_303 = builder.add_for(body_244, indvar_303, condition_303, init_303, update_303);
    auto& body_303 = loop_303.root();

    // 303_block
    {
        auto subset_in = symbolic::
            add(indvar_11,
                symbolic::
                    add(indvar_303,
                        symbolic::
                            add(symbolic::mul(symbolic::symbol("_0"), symbolic::symbol("_11")),
                                symbolic::mul(symbolic::symbol("_0"), symbolic::symbol("_19")))));

        auto& block_303_1 = builder.add_block(body_303);
        auto& _1_in = builder.add_access(block_303_1, "_1");
        auto& _330_out = builder.add_access(block_303_1, "_330");
        auto& tasklet = builder.add_tasklet(block_303_1, data_flow::TaskletCode::neg, "_out", {"_in"});
        builder.add_computational_memlet(block_303_1, _1_in, tasklet, "_in", {subset_in}, edge_desc);
        builder.add_computational_memlet(block_303_1, tasklet, "_out", _330_out, {});

        auto subset_in_2 = symbolic::
            add(indvar_11,
                symbolic::
                    add(indvar_244,
                        symbolic::
                            add(symbolic::mul(symbolic::symbol("_0"), symbolic::symbol("_11")),
                                symbolic::mul(symbolic::symbol("_0"), symbolic::symbol("_303")))));

        auto subset_out = symbolic::
            add(indvar_11,
                symbolic::
                    add(indvar_244,
                        symbolic::
                            add(symbolic::mul(symbolic::symbol("_0"), symbolic::symbol("_11")),
                                symbolic::mul(symbolic::symbol("_0"), symbolic::symbol("_19")))));

        auto& block_303_2 = builder.add_block(body_303);
        auto& _1_in_2 = builder.add_access(block_303_2, "_1");
        auto& _1_in_3 = builder.add_access(block_303_2, "_1");
        auto& _330_in_2 = builder.add_access(block_303_2, "_330");
        auto& _1_out_2 = builder.add_access(block_303_2, "_1");
        auto& tasklet_2 =
            builder.add_tasklet(block_303_2, data_flow::TaskletCode::fma, "_out", {"_in0", "_in1", "_in2"});
        builder.add_computational_memlet(block_303_2, _1_in_2, tasklet_2, "_in0", {subset_in_2}, edge_desc);
        builder.add_computational_memlet(block_303_2, _1_in_3, tasklet_2, "_in1", {subset_out}, edge_desc);
        builder.add_computational_memlet(block_303_2, _330_in_2, tasklet_2, "_in2", {});
        builder.add_computational_memlet(block_303_2, tasklet_2, "_out", _1_out_2, {subset_out}, edge_desc);
    }

    // block _19
    {
        auto subset = symbolic::
            add(indvar_11,
                symbolic::
                    add(indvar_19,
                        symbolic::
                            add(symbolic::mul(symbolic::symbol("_0"), symbolic::symbol("_11")),
                                symbolic::mul(symbolic::symbol("_0"), symbolic::symbol("_19")))));

        auto& block_19 = builder.add_block(body_19);
        auto& _1_in = builder.add_access(block_19, "_1");
        auto& _258_out = builder.add_access(block_19, "_258");
        auto& tasklet = builder.add_tasklet(block_19, data_flow::TaskletCode::mul, "_out", {"1.0f", "_in1"});
        builder.add_computational_memlet(block_19, _1_in, tasklet, "_in1", {subset}, edge_desc);
        builder.add_computational_memlet(block_19, tasklet, "_out", _258_out, {});
    }

    // _260 loop
    auto bound_260 = symbolic::integer(16);
    auto indvar_260 = symbolic::symbol("_260");
    auto init_260 = symbolic::add(indvar_19, symbolic::one());
    auto condition_260 = symbolic::Lt(indvar_260, bound_260);
    auto update_260 = symbolic::add(indvar_260, symbolic::integer(1));

    auto& loop_260 = builder.add_for(body_19, indvar_260, condition_260, init_260, update_260);
    auto& body_260 = loop_260.root();

    // _267 loop
    auto bound_267 = indvar_19;
    auto indvar_267 = symbolic::symbol("_267");
    auto init_267 = symbolic::zero();
    auto condition_267 = symbolic::Lt(indvar_267, bound_267);
    auto update_267 = symbolic::add(indvar_267, symbolic::integer(1));

    auto& loop_267 = builder.add_for(body_260, indvar_267, condition_267, init_267, update_267);
    auto& body_267 = loop_267.root();

    // block _267
    {
        auto subset_in_1 = symbolic::
            add(indvar_11,
                symbolic::
                    add(indvar_267,
                        symbolic::
                            add(symbolic::mul(symbolic::symbol("_0"), symbolic::symbol("_11")),
                                symbolic::mul(symbolic::symbol("_0"), symbolic::symbol("_260")))));

        auto subset_in_2 = symbolic::
            add(indvar_11,
                symbolic::
                    add(indvar_19,
                        symbolic::
                            add(symbolic::mul(symbolic::symbol("_0"), symbolic::symbol("_11")),
                                symbolic::mul(symbolic::symbol("_0"), symbolic::symbol("_267")))));

        auto subset_in_3 = symbolic::
            add(indvar_11,
                symbolic::
                    add(indvar_19,
                        symbolic::
                            add(symbolic::mul(symbolic::symbol("_0"), symbolic::symbol("_11")),
                                symbolic::mul(symbolic::symbol("_0"), symbolic::symbol("_260")))));

        auto& block_267_1 = builder.add_block(body_267);
        auto& _1_in_1 = builder.add_access(block_267_1, "_1");
        auto& _298_out = builder.add_access(block_267_1, "_298");
        auto& tasklet = builder.add_tasklet(block_267_1, data_flow::TaskletCode::neg, "_out", {"_in"});
        builder.add_computational_memlet(block_267_1, _1_in_1, tasklet, "_in", {subset_in_1}, edge_desc);
        builder.add_computational_memlet(block_267_1, tasklet, "_out", _298_out, {});

        auto& block_267_2 = builder.add_block(body_267);
        auto& _1_in_2 = builder.add_access(block_267_2, "_1");
        auto& _1_in_3 = builder.add_access(block_267_2, "_1");
        auto& _298_in_2 = builder.add_access(block_267_2, "_298");
        auto& _1_out_2 = builder.add_access(block_267_2, "_1");
        auto& tasklet_2 =
            builder.add_tasklet(block_267_2, data_flow::TaskletCode::fma, "_out", {"_in0", "_in1", "_in2"});
        builder.add_computational_memlet(block_267_2, _298_in_2, tasklet_2, "_in0", {});
        builder.add_computational_memlet(block_267_2, _1_in_2, tasklet_2, "_in1", {subset_in_2}, edge_desc);
        builder.add_computational_memlet(block_267_2, _1_in_3, tasklet_2, "_in2", {subset_in_3}, edge_desc);
        builder.add_computational_memlet(block_267_2, tasklet_2, "_out", _1_out_2, {subset_in_3}, edge_desc);
    }

    // block _260
    {
        auto subset = symbolic::
            add(indvar_11,
                symbolic::
                    add(indvar_19,
                        symbolic::
                            add(symbolic::mul(symbolic::symbol("_0"), symbolic::symbol("_11")),
                                symbolic::mul(symbolic::symbol("_0"), symbolic::symbol("_260")))));

        auto& block_260 = builder.add_block(body_260);
        auto& _1_in = builder.add_access(block_260, "_1");
        auto& _1_out = builder.add_access(block_260, "_1");
        auto& _258_in = builder.add_access(block_260, "_258");
        auto& tasklet = builder.add_tasklet(block_260, data_flow::TaskletCode::mul, "_out", {"_in0", "_in1"});
        builder.add_computational_memlet(block_260, _258_in, tasklet, "_in0", {});
        builder.add_computational_memlet(block_260, _1_in, tasklet, "_in1", {subset}, edge_desc);
        builder.add_computational_memlet(block_260, tasklet, "_out", _1_out, {subset}, edge_desc);
    }

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();
    auto& dependencies_11 = analysis.dependencies(loop_11);
    auto& dependencies_19 = analysis.dependencies(loop_19);
    auto& dependencies_244 = analysis.dependencies(loop_244);
    auto& dependencies_303 = analysis.dependencies(loop_303);
    auto& dependencies_260 = analysis.dependencies(loop_260);
    auto& dependencies_267 = analysis.dependencies(loop_267);

    // Check
    EXPECT_EQ(dependencies_11.size(), 9);
    EXPECT_EQ(dependencies_11.at("_1"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_READ_WRITE);
    EXPECT_EQ(dependencies_11.at("_330"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
    EXPECT_EQ(dependencies_11.at("_19"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
    EXPECT_EQ(dependencies_11.at("_244"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
    EXPECT_EQ(dependencies_11.at("_303"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
    EXPECT_EQ(dependencies_11.at("_258"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
    EXPECT_EQ(dependencies_11.at("_260"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
    EXPECT_EQ(dependencies_11.at("_267"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
    EXPECT_EQ(dependencies_11.at("_298"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);

    EXPECT_EQ(dependencies_19.size(), 8);
    EXPECT_EQ(dependencies_19.at("_1"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_READ_WRITE);
    EXPECT_EQ(dependencies_19.at("_330"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
    EXPECT_EQ(dependencies_19.at("_244"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
    EXPECT_EQ(dependencies_19.at("_303"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
    EXPECT_EQ(dependencies_19.at("_258"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
    EXPECT_EQ(dependencies_19.at("_260"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
    EXPECT_EQ(dependencies_19.at("_267"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
    EXPECT_EQ(dependencies_19.at("_298"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);

    EXPECT_EQ(dependencies_244.size(), 2);
    EXPECT_EQ(dependencies_244.at("_330"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
    EXPECT_EQ(dependencies_244.at("_303"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);

    EXPECT_EQ(dependencies_303.size(), 2);
    EXPECT_EQ(dependencies_303.at("_1"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_READ_WRITE);
    EXPECT_EQ(dependencies_303.at("_330"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);

    EXPECT_EQ(dependencies_267.size(), 2);
    EXPECT_EQ(dependencies_267.at("_1"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_READ_WRITE);
    EXPECT_EQ(dependencies_267.at("_298"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);

    EXPECT_EQ(dependencies_260.size(), 2);
    EXPECT_EQ(dependencies_260.at("_267"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
    EXPECT_EQ(dependencies_260.at("_298"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
}
