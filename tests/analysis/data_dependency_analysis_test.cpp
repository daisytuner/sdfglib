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

    builder.add_container("A", types::Scalar(types::PrimitiveType::Int32));

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& output_node = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                        {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                        {{"0", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {});

    // Run analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    auto& analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;

    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    symbolic::Assumptions assumptions = assumptions_analysis.get(block, true);
    analysis.visit_block(users, assumptions_analysis, assumptions, block, undefined,
                         open_definitions, closed_definitions);

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
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                        {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                        {{"0", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {symbolic::integer(0)});

    // Run analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    auto& analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;

    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    symbolic::Assumptions assumptions = assumptions_analysis.get(block, true);
    analysis.visit_block(users, assumptions_analysis, assumptions, block, undefined,
                         open_definitions, closed_definitions);

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

    builder.add_container("A", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("B", types::Scalar(types::PrimitiveType::Int32));

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& input_node = builder.add_access(block, "A");
    auto& output_node = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                        {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                        {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {});
    builder.add_memlet(block, input_node, "void", tasklet, "_in", {});

    // Run analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    auto& analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;

    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    symbolic::Assumptions assumptions = assumptions_analysis.get(block, true);
    analysis.visit_block(users, assumptions_analysis, assumptions, block, undefined,
                         open_definitions, closed_definitions);

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
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                        {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                        {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {symbolic::integer(0)});
    builder.add_memlet(block, input_node, "void", tasklet, "_in", {symbolic::integer(0)});

    // Run analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    auto& analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;

    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    symbolic::Assumptions assumptions = assumptions_analysis.get(block, true);
    analysis.visit_block(users, assumptions_analysis, assumptions, block, undefined,
                         open_definitions, closed_definitions);

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
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                        {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                        {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {symbolic::integer(0)});
    builder.add_memlet(block, input_node, "void", tasklet, "_in", {symbolic::integer(0)});

    auto& output_node2 = builder.add_access(block, "C");
    auto& tasklet2 = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                         {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                         {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block, tasklet2, "_out", output_node2, "void", {symbolic::integer(0)});
    builder.add_memlet(block, output_node, "void", tasklet2, "_in", {symbolic::integer(1)});

    // Run analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    auto& analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;

    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    symbolic::Assumptions assumptions = assumptions_analysis.get(block, true);
    analysis.visit_block(users, assumptions_analysis, assumptions, block, undefined,
                         open_definitions, closed_definitions);

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
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& output_node = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                        {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                        {{"0", types::Scalar(types::PrimitiveType::Int32)}});
    auto& memlet =
        builder.add_memlet(block, tasklet, "_out", output_node, "void", {symbolic::symbol("i")});

    // Run analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    auto& analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;

    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    symbolic::Assumptions assumptions = assumptions_analysis.get(block, true);
    analysis.visit_block(users, assumptions_analysis, assumptions, block, undefined,
                         open_definitions, closed_definitions);

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

    builder.add_container("A", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("B", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("C", types::Scalar(types::PrimitiveType::Int32));

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& input_node = builder.add_access(block, "A");
    auto& output_node = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                        {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                        {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {});
    builder.add_memlet(block, input_node, "void", tasklet, "_in", {});

    auto& output_node2 = builder.add_access(block, "C");
    auto& tasklet2 = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                         {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                         {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block, tasklet2, "_out", output_node2, "void", {});
    builder.add_memlet(block, output_node, "void", tasklet2, "_in", {});

    // Run analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    auto& analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;

    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    symbolic::Assumptions assumptions = assumptions_analysis.get(block, true);
    analysis.visit_block(users, assumptions_analysis, assumptions, block, undefined,
                         open_definitions, closed_definitions);

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
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                        {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                        {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {symbolic::integer(0)});
    builder.add_memlet(block, input_node, "void", tasklet, "_in", {symbolic::integer(0)});

    auto& output_node2 = builder.add_access(block, "C");
    auto& tasklet2 = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                         {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                         {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block, tasklet2, "_out", output_node2, "void", {symbolic::integer(0)});
    builder.add_memlet(block, output_node, "void", tasklet2, "_in", {symbolic::integer(0)});

    // Run analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    auto& analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;

    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    symbolic::Assumptions assumptions = assumptions_analysis.get(block, true);
    analysis.visit_block(users, assumptions_analysis, assumptions, block, undefined,
                         open_definitions, closed_definitions);

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
    auto& analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;

    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    symbolic::Assumptions assumptions = assumptions_analysis.get(root, true);
    analysis.visit_sequence(users, assumptions_analysis, assumptions, root, undefined,
                            open_definitions, closed_definitions);

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
    auto& analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;

    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    symbolic::Assumptions assumptions = assumptions_analysis.get(root, true);
    analysis.visit_sequence(users, assumptions_analysis, assumptions, root, undefined,
                            open_definitions, closed_definitions);

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
    auto& analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;

    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    symbolic::Assumptions assumptions = assumptions_analysis.get(root, true);
    analysis.visit_sequence(users, assumptions_analysis, assumptions, root, undefined,
                            open_definitions, closed_definitions);

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
    auto& tasklet = builder.add_tasklet(block1, data_flow::TaskletCode::assign,
                                        {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                        {{"0", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block1, tasklet, "_out", output_node, "void", {symbolic::integer(0)});

    auto& block2 = builder.add_block(root);
    auto& output_node2 = builder.add_access(block2, "A");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign,
                                         {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                         {{"1", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block2, tasklet2, "_out", output_node2, "void", {symbolic::integer(0)});

    // Run analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    auto& analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;

    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    symbolic::Assumptions assumptions = assumptions_analysis.get(root, true);
    analysis.visit_sequence(users, assumptions_analysis, assumptions, root, undefined,
                            open_definitions, closed_definitions);

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
    auto& tasklet = builder.add_tasklet(block1, data_flow::TaskletCode::assign,
                                        {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                        {{"0", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block1, tasklet, "_out", output_node, "void", {symbolic::integer(0)});

    auto& block2 = builder.add_block(root);
    auto& output_node2 = builder.add_access(block2, "A");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign,
                                         {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                         {{"1", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block2, tasklet2, "_out", output_node2, "void", {symbolic::integer(1)});

    // Run analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    auto& analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;

    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    symbolic::Assumptions assumptions = assumptions_analysis.get(root, true);
    analysis.visit_sequence(users, assumptions_analysis, assumptions, root, undefined,
                            open_definitions, closed_definitions);

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

/*

TEST(DataDependencyAnalysisTest, visit_for) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    builder.add_container("B", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));

    auto& root = builder.subject().root();

    auto& for_loop = builder.add_for(
        root, symbolic::symbol("i"), symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0), symbolic::add(symbolic::symbol("i"), symbolic::integer(1)));

    auto& block = builder.add_block(for_loop.root());
    auto& input_node = builder.add_access(block, "i");
    auto& output_node = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                        {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                        {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {});
    builder.add_memlet(block, input_node, "void", tasklet, "_in", {});

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
    symbolic::Assumptions assumptions = assumptions_analysis.get(for_loop, true);
    analysis.visit_for(users, assumptions_analysis, assumptions, for_loop, undefined,
                       open_definitions, closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 0);
    EXPECT_EQ(open_definitions.size(), 3);
    EXPECT_EQ(closed_definitions.size(), 0);

    bool foundB = false;
    int both_i = 0;

    for (auto entry : open_definitions) {
        if (entry.first->container() == "B") {
            foundB = true;
            EXPECT_EQ(entry.first->use(), analysis::Use::WRITE);
            EXPECT_EQ(entry.first->container(), "B");
            EXPECT_EQ(entry.first->element(), &output_node);
            EXPECT_EQ(entry.second.size(), 0);
        } else if (entry.first->container() == "i") {
            both_i++;
            EXPECT_EQ(entry.first->use(), analysis::Use::WRITE);
            EXPECT_EQ(entry.first->container(), "i");
            EXPECT_EQ(entry.first->element(), &for_loop);
            EXPECT_EQ(entry.second.size(), 3);
        }
    }

    EXPECT_TRUE(foundB);
    EXPECT_EQ(both_i, 2);
}

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
                                        {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                        {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {});
    builder.add_memlet(block, input_node, "void", tasklet, "_in", {});

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
                                        {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                        {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {});
    builder.add_memlet(block, input_node, "void", tasklet, "_in", {});

    auto& output_node2 = builder.add_access(block, "A");
    auto& tasklet2 = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                         {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                         {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block, tasklet2, "_out", output_node2, "void", {});
    builder.add_memlet(block, output_node, "void", tasklet2, "_in", {});

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

TEST(DataDependencyAnalysisTest, visit_if_else_complete) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    builder.add_container("A", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("B", types::Scalar(types::PrimitiveType::Int32));

    auto& root = builder.subject().root();

    auto& if_else = builder.add_if_else(root);
    auto& true_case =
        builder.add_case(if_else, symbolic::Le(symbolic::symbol("A"), symbolic::integer(0)));
    auto& false_case =
        builder.add_case(if_else, symbolic::Gt(symbolic::symbol("A"), symbolic::integer(0)));

    auto& block = builder.add_block(true_case);
    auto& input_node = builder.add_access(block, "A");
    auto& output_node = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                        {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                        {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {});
    builder.add_memlet(block, input_node, "void", tasklet, "_in", {});

    auto& block2 = builder.add_block(false_case);
    auto& input_node2 = builder.add_access(block2, "B");
    auto& output_node2 = builder.add_access(block2, "A");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign,
                                         {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                         {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block2, tasklet2, "_out", output_node2, "void", {});
    builder.add_memlet(block2, input_node2, "void", tasklet2, "_in", {});

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
    symbolic::Assumptions assumptions = assumptions_analysis.get(if_else, true);
    analysis.visit_if_else(users, assumptions_analysis, assumptions, if_else, undefined,
                           open_definitions, closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 3);
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
            EXPECT_EQ(entry.second.size(), 0);
        } else if (entry.first->container() == "B") {
            foundB = true;
            EXPECT_EQ(entry.first->use(), analysis::Use::WRITE);
            EXPECT_EQ(entry.first->container(), "B");
            EXPECT_EQ(entry.first->element(), &output_node);
            EXPECT_EQ(entry.second.size(), 0);
        }
    }

    EXPECT_TRUE(foundA && foundB);

    int A_count = 0;
    foundA = false;
    foundB = false;

    for (auto entry : undefined) {
        if (entry->container() == "A") {
            EXPECT_EQ(entry->use(), analysis::Use::READ);
            EXPECT_EQ(entry->container(), "A");
            if (entry->element() == &input_node) {
                foundA = true;
            } else {
                EXPECT_EQ(entry->element(), &if_else);
                A_count++;
            }
        } else if (entry->container() == "B") {
            EXPECT_EQ(entry->use(), analysis::Use::READ);
            EXPECT_EQ(entry->container(), "B");
            EXPECT_EQ(entry->element(), &input_node2);
            foundB = true;
        }
    }

    EXPECT_EQ(A_count, 1);
    EXPECT_TRUE(foundA && foundB);
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
                                        {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                        {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {});
    builder.add_memlet(block, input_node, "void", tasklet, "_in", {});

    auto& for_loop = builder.add_for(
        root, symbolic::symbol("i"), symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0), symbolic::add(symbolic::symbol("i"), symbolic::integer(1)));

    auto& block2 = builder.add_block(for_loop.root());
    auto& input_node2 = builder.add_access(block2, "i");
    auto& output_node2 = builder.add_access(block2, "A");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign,
                                         {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                         {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block2, tasklet2, "_out", output_node2, "void", {});
    builder.add_memlet(block2, input_node2, "void", tasklet2, "_in", {});

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
                                        {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                        {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {});
    builder.add_memlet(block, input_node, "void", tasklet, "_in", {});

    auto& while_loop = builder.add_while(root);

    auto& block2 = builder.add_block(while_loop.root());
    auto& input_node2 = builder.add_access(block2, "B");
    auto& output_node2 = builder.add_access(block2, "A");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign,
                                         {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                         {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block2, tasklet2, "_out", output_node2, "void", {});
    builder.add_memlet(block2, input_node2, "void", tasklet2, "_in", {});

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

TEST(DataDependencyAnalysisTest, visit_sequence_if_else_complete) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    builder.add_container("A", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("B", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("C", types::Scalar(types::PrimitiveType::Int32));

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root, {{symbolic::symbol("C"), symbolic::symbol("B")}});
    auto& input_node = builder.add_access(block, "A");
    auto& output_node = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                        {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                        {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {});
    builder.add_memlet(block, input_node, "void", tasklet, "_in", {});

    auto& if_else = builder.add_if_else(root);

    auto& case1 =
        builder.add_case(if_else, symbolic::Lt(symbolic::symbol("B"), symbolic::integer(10)));

    auto& block2 = builder.add_block(case1);
    auto& input_node2 = builder.add_access(block2, "B");
    auto& output_node2 = builder.add_access(block2, "C");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign,
                                         {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                         {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block2, tasklet2, "_out", output_node2, "void", {});
    builder.add_memlet(block2, input_node2, "void", tasklet2, "_in", {});

    auto& case2 =
        builder.add_case(if_else, symbolic::Le(symbolic::integer(10), symbolic::symbol("B")));

    auto& block3 = builder.add_block(case2);
    auto& input_node3 = builder.add_access(block3, "B");
    auto& output_node3 = builder.add_access(block3, "C");
    auto& tasklet3 = builder.add_tasklet(block3, data_flow::TaskletCode::assign,
                                         {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                         {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block3, tasklet3, "_out", output_node3, "void", {});
    builder.add_memlet(block3, input_node3, "void", tasklet3, "_in", {});

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
    EXPECT_EQ(open_definitions.size(), 3);
    EXPECT_EQ(closed_definitions.size(), 1);

    auto read = *undefined.begin();

    EXPECT_EQ(read->use(), analysis::Use::READ);
    EXPECT_EQ(read->container(), "A");
    EXPECT_EQ(read->element(), &input_node);

    bool foundB = false;
    bool foundB_left = false;
    bool foundB_right = false;
    int countB_cond = 0;
    bool foundB_trans = false;
    bool foundC_left = false;
    bool foundC_right = false;

    for (auto entry : open_definitions) {
        if (entry.first->container() == "B") {
            foundB = true;
            EXPECT_EQ(entry.first->use(), analysis::Use::WRITE);
            EXPECT_EQ(entry.first->container(), "B");
            EXPECT_EQ(entry.first->element(), &output_node);
            EXPECT_EQ(entry.second.size(), 4);
            for (auto entry2 : entry.second) {
                if (entry2->element() == &input_node2) {
                    foundB_left = true;
                    EXPECT_EQ(entry2->use(), analysis::Use::READ);
                    EXPECT_EQ(entry2->container(), "B");
                } else if (entry2->element() == &input_node3) {
                    foundB_right = true;
                    EXPECT_EQ(entry2->use(), analysis::Use::READ);
                    EXPECT_EQ(entry2->container(), "B");
                } else if (entry2->element() == &if_else) {
                    countB_cond++;
                    EXPECT_EQ(entry2->use(), analysis::Use::READ);
                    EXPECT_EQ(entry2->container(), "B");
                } else if (entry2->element() == &builder_opt.subject().root().at(0).second) {
                    foundB_trans = true;
                    EXPECT_EQ(entry2->use(), analysis::Use::READ);
                    EXPECT_EQ(entry2->container(), "B");
                }
            }
        } else if (entry.first->container() == "C") {
            EXPECT_EQ(entry.first->use(), analysis::Use::WRITE);
            if (entry.first->element() == &output_node2) {
                EXPECT_EQ(entry.second.size(), 0);
                foundC_left = true;
            } else if (entry.first->element() == &output_node3) {
                EXPECT_EQ(entry.second.size(), 0);
                foundC_right = true;
            }
        }
    }

    EXPECT_EQ(countB_cond, 1);
    EXPECT_TRUE(foundB && foundB_left && foundB_right && foundB_trans);
    EXPECT_TRUE(foundC_left && foundC_right);

    auto closed = *closed_definitions.begin();

    EXPECT_EQ(closed.first->use(), analysis::Use::WRITE);
    EXPECT_EQ(closed.first->container(), "C");
    EXPECT_EQ(closed.first->element(), &builder_opt.subject().root().at(0).second);
    EXPECT_EQ(closed.second.size(), 0);
}

TEST(DataDependencyAnalysisTest, visit_sequence_if_else_incomplete) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    builder.add_container("A", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("B", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("C", types::Scalar(types::PrimitiveType::Int32));

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root, {{symbolic::symbol("C"), symbolic::symbol("B")}});
    auto& input_node = builder.add_access(block, "A");
    auto& output_node = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                        {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                        {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {});
    builder.add_memlet(block, input_node, "void", tasklet, "_in", {});

    auto& if_else = builder.add_if_else(root);

    auto& case1 =
        builder.add_case(if_else, symbolic::Lt(symbolic::symbol("B"), symbolic::integer(10)));

    auto& block2 = builder.add_block(case1);
    auto& input_node2 = builder.add_access(block2, "B");
    auto& output_node2 = builder.add_access(block2, "C");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign,
                                         {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                         {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block2, tasklet2, "_out", output_node2, "void", {});
    builder.add_memlet(block2, input_node2, "void", tasklet2, "_in", {});

    auto& case2 =
        builder.add_case(if_else, symbolic::Lt(symbolic::integer(10), symbolic::symbol("B")));

    auto& block3 = builder.add_block(case2);
    auto& input_node3 = builder.add_access(block3, "B");
    auto& output_node3 = builder.add_access(block3, "C");
    auto& tasklet3 = builder.add_tasklet(block3, data_flow::TaskletCode::assign,
                                         {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                         {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block3, tasklet3, "_out", output_node3, "void", {});
    builder.add_memlet(block3, input_node3, "void", tasklet3, "_in", {});

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
    bool foundB_left = false;
    bool foundB_right = false;
    int countB_cond = 0;
    bool foundB_trans = false;
    bool foundC = false;
    bool foundC_left = false;
    bool foundC_right = false;

    for (auto entry : open_definitions) {
        if (entry.first->container() == "B") {
            foundB = true;
            EXPECT_EQ(entry.first->use(), analysis::Use::WRITE);
            EXPECT_EQ(entry.first->container(), "B");
            EXPECT_EQ(entry.first->element(), &output_node);
            EXPECT_EQ(entry.second.size(), 4);
            for (auto entry2 : entry.second) {
                if (entry2->element() == &input_node2) {
                    foundB_left = true;
                    EXPECT_EQ(entry2->use(), analysis::Use::READ);
                    EXPECT_EQ(entry2->container(), "B");
                } else if (entry2->element() == &input_node3) {
                    foundB_right = true;
                    EXPECT_EQ(entry2->use(), analysis::Use::READ);
                    EXPECT_EQ(entry2->container(), "B");
                } else if (entry2->element() == &if_else) {
                    countB_cond++;
                    EXPECT_EQ(entry2->use(), analysis::Use::READ);
                    EXPECT_EQ(entry2->container(), "B");
                } else if (entry2->element() == &builder_opt.subject().root().at(0).second) {
                    foundB_trans = true;
                    EXPECT_EQ(entry2->use(), analysis::Use::READ);
                    EXPECT_EQ(entry2->container(), "B");
                }
            }
        } else if (entry.first->container() == "C") {
            EXPECT_EQ(entry.first->use(), analysis::Use::WRITE);
            if (entry.first->element() == &output_node2) {
                EXPECT_EQ(entry.second.size(), 0);
                foundC_left = true;
            } else if (entry.first->element() == &output_node3) {
                EXPECT_EQ(entry.second.size(), 0);
                foundC_right = true;
            } else if (entry.first->element() == &builder_opt.subject().root().at(0).second) {
                EXPECT_EQ(entry.second.size(), 0);
                foundC = true;
            }
        }
    }

    EXPECT_EQ(countB_cond, 1);
    EXPECT_TRUE(foundB && foundB_left && foundB_right && foundB_trans);
    EXPECT_TRUE(foundC_left && foundC_right);
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
                                        {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                        {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {});
    builder.add_memlet(block, input_node, "void", tasklet, "_in", {});

    auto& input_node3 = output_node;
    auto& output_node3 = builder.add_access(block, "C");
    auto& tasklet3 = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                         {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                         {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block, tasklet3, "_out", output_node3, "void", {});
    builder.add_memlet(block, input_node3, "void", tasklet3, "_in", {});

    auto& input_node2 = output_node3;
    auto& output_node2 = builder.add_access(block, "B");
    auto& tasklet2 = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                         {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                         {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block, tasklet2, "_out", output_node2, "void", {});
    builder.add_memlet(block, input_node2, "void", tasklet2, "_in", {});

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
        builder.add_memlet(outer_block, outer_input_node, "void", outer_output_node, "refs", {sym});

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
