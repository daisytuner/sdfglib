#include "sdfg/analysis/data_parallelism_analysis.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/array.h"

using namespace sdfg;

TEST(TestDataParallelism, Substitution_Affine) {
    symbolic::SymbolicMap replacements;
    std::vector<std::string> substitutions;
    auto res = analysis::DataParallelismAnalysis::substitution(
        {symbolic::add(symbolic::mul(symbolic::symbol("N"), symbolic::symbol("i")),
                       symbolic::symbol("j"))},
        {symbolic::add(symbolic::mul(symbolic::symbol("N"), symbolic::symbol("i")),
                       symbolic::symbol("j"))},
        "j", {}, replacements, substitutions);

    EXPECT_EQ(substitutions.size(), 1);
    EXPECT_EQ(substitutions.at(0), "c_0");
    EXPECT_EQ(replacements.size(), 1);

    auto p = symbolic::mul(symbolic::symbol("N"), symbolic::symbol("i"));
    EXPECT_TRUE(symbolic::eq(replacements.at(p), symbolic::symbol("c_0")));

    EXPECT_TRUE(symbolic::eq(res.first.at(0),
                             symbolic::add(symbolic::symbol("c_0"), symbolic::symbol("j"))));
    EXPECT_TRUE(symbolic::eq(res.second.at(0),
                             symbolic::add(symbolic::symbol("c_0"), symbolic::symbol("j"))));
}

TEST(TestDataParallelism, Delinearization_Basic) {
    symbolic::Assumptions assumptions;

    symbolic::Assumption a1(symbolic::symbol("N"));
    a1.lower_bound(symbolic::integer(0));
    a1.upper_bound(symbolic::infty(1));
    assumptions.insert({a1.symbol(), a1});

    symbolic::Assumption a2(symbolic::symbol("M"));
    a2.lower_bound(symbolic::integer(0));
    a2.upper_bound(symbolic::infty(1));
    assumptions.insert({a2.symbol(), a2});

    symbolic::Assumption a3(symbolic::symbol("i"));
    a3.lower_bound(symbolic::integer(0));
    a3.upper_bound(symbolic::symbol("M"));
    assumptions.insert({a3.symbol(), a3});

    symbolic::Assumption a4(symbolic::symbol("j"));
    a4.lower_bound(symbolic::integer(0));
    a4.upper_bound(symbolic::sub(symbolic::symbol("N"), symbolic::integer(1)));
    assumptions.insert({a4.symbol(), a4});

    auto res = analysis::DataParallelismAnalysis::delinearization(
        {symbolic::add(symbolic::mul(symbolic::symbol("N"), symbolic::symbol("i")),
                       symbolic::symbol("j"))},
        {symbolic::add(symbolic::mul(symbolic::symbol("N"), symbolic::symbol("i")),
                       symbolic::symbol("j"))},
        {"j", "i"}, assumptions);

    EXPECT_EQ(res.first.size(), 2);
    EXPECT_TRUE(symbolic::eq(res.first.at(0), symbolic::symbol("i")));
    EXPECT_TRUE(symbolic::eq(res.first.at(1), symbolic::symbol("j")));
    EXPECT_EQ(res.second.size(), 2);
    EXPECT_TRUE(symbolic::eq(res.second.at(0), symbolic::symbol("i")));
    EXPECT_TRUE(symbolic::eq(res.second.at(1), symbolic::symbol("j")));
}

TEST(TestDataParallelism, Delinearization_Equality) {
    symbolic::Assumptions assumptions;

    symbolic::Assumption a1(symbolic::symbol("N"));
    a1.lower_bound(symbolic::integer(0));
    a1.upper_bound(symbolic::infty(1));
    assumptions.insert({a1.symbol(), a1});

    symbolic::Assumption a2(symbolic::symbol("M"));
    a2.lower_bound(symbolic::integer(0));
    a2.upper_bound(symbolic::infty(1));
    assumptions.insert({a2.symbol(), a2});

    symbolic::Assumption a3(symbolic::symbol("i"));
    a3.lower_bound(symbolic::integer(0));
    a3.upper_bound(symbolic::symbol("M"));
    assumptions.insert({a3.symbol(), a3});

    symbolic::Assumption a4(symbolic::symbol("j"));
    a4.lower_bound(symbolic::integer(0));
    a4.upper_bound(symbolic::symbol("N"));
    assumptions.insert({a4.symbol(), a4});

    auto res = analysis::DataParallelismAnalysis::delinearization(
        {symbolic::add(symbolic::mul(symbolic::symbol("N"), symbolic::symbol("i")),
                       symbolic::symbol("j"))},
        {symbolic::add(symbolic::mul(symbolic::symbol("N"), symbolic::symbol("i")),
                       symbolic::symbol("j"))},
        {"j", "i"}, assumptions);

    EXPECT_EQ(res.first.size(), 1);
    EXPECT_EQ(res.second.size(), 1);
}

TEST(TestDataParallelism, Delinearization_Negative) {
    symbolic::Assumptions assumptions;

    symbolic::Assumption a1(symbolic::symbol("N"));
    a1.lower_bound(symbolic::integer(0));
    a1.upper_bound(symbolic::infty(1));
    assumptions.insert({a1.symbol(), a1});

    symbolic::Assumption a2(symbolic::symbol("M"));
    a2.lower_bound(symbolic::integer(0));
    a2.upper_bound(symbolic::infty(1));
    assumptions.insert({a2.symbol(), a2});

    symbolic::Assumption a3(symbolic::symbol("i"));
    a3.lower_bound(symbolic::integer(0));
    a3.upper_bound(symbolic::symbol("M"));
    assumptions.insert({a3.symbol(), a3});

    symbolic::Assumption a4(symbolic::symbol("j"));
    a4.lower_bound(symbolic::integer(-1));
    a4.upper_bound(symbolic::sub(symbolic::symbol("N"), symbolic::integer(1)));
    assumptions.insert({a4.symbol(), a4});

    auto res = analysis::DataParallelismAnalysis::delinearization(
        {symbolic::add(symbolic::mul(symbolic::symbol("N"), symbolic::symbol("i")),
                       symbolic::symbol("j"))},
        {symbolic::add(symbolic::mul(symbolic::symbol("N"), symbolic::symbol("i")),
                       symbolic::symbol("j"))},
        {"j", "i"}, assumptions);

    EXPECT_EQ(res.first.size(), 1);
    EXPECT_EQ(res.second.size(), 1);
}

TEST(TestDataParallelism, Trivial_1D_Identity) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
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
    auto& a2 = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", base_desc},
                                        {{"_in", base_desc}});
    builder.add_memlet(block, a1, "void", tasklet, "_in", {symbolic::integer(0)});
    builder.add_memlet(block, tasklet, "_out", a2, "void", {symbolic::integer(0)});

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::DataParallelismAnalysis>();
    auto& dependencies = analysis.get(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 1);
    EXPECT_EQ(dependencies.at("A"), analysis::Parallelism::REDUCTION);
}

TEST(TestDataParallelism, Reduction) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
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
    auto& a2 = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::add, {"_out", base_desc},
                                        {{"_in", base_desc}, {"1", base_desc}});
    builder.add_memlet(block, a1, "void", tasklet, "_in", {symbolic::integer(0)});
    builder.add_memlet(block, tasklet, "_out", a2, "void", {symbolic::integer(0)});

    auto& graph = block.dataflow();
    EXPECT_EQ(graph.nodes().size(), 3);
    EXPECT_EQ(graph.edges().size(), 2);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::DataParallelismAnalysis>();
    auto& dependencies = analysis.get(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 1);
    EXPECT_EQ(dependencies.at("A"), analysis::Parallelism::REDUCTION);
}

TEST(TestDataParallelism, Map_1D_WriteOnly) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
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
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", base_desc},
                                        {{"0", base_desc}});
    builder.add_memlet(block, tasklet, "_out", A, "void", {indvar});

    auto& graph = block.dataflow();
    EXPECT_EQ(graph.nodes().size(), 2);
    EXPECT_EQ(graph.edges().size(), 1);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::DataParallelismAnalysis>();
    auto& dependencies = analysis.get(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 1);
    EXPECT_EQ(dependencies.at("A"), analysis::Parallelism::PARALLEL);
}

TEST(TestDataParallelism, Map_1D_Scalar) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("a", base_desc, true);
    builder.add_container("A", desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
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
    auto& a = builder.add_access(block, "a");
    auto& A = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", base_desc},
                                        {{"_in", base_desc}});
    builder.add_memlet(block, a, "void", tasklet, "_in", {});
    builder.add_memlet(block, tasklet, "_out", A, "void", {indvar});

    auto& graph = block.dataflow();
    EXPECT_EQ(graph.nodes().size(), 3);
    EXPECT_EQ(graph.edges().size(), 2);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::DataParallelismAnalysis>();
    auto& dependencies = analysis.get(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 2);
    EXPECT_EQ(dependencies.at("a"), analysis::Parallelism::READONLY);
    EXPECT_EQ(dependencies.at("A"), analysis::Parallelism::PARALLEL);
}

TEST(TestDataParallelism, Map_1D_Identity) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
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
    auto& A1 = builder.add_access(block, "A");
    auto& A2 = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", base_desc},
                                        {{"_in", base_desc}});
    builder.add_memlet(block, A1, "void", tasklet, "_in", {indvar});
    builder.add_memlet(block, tasklet, "_out", A2, "void", {indvar});

    auto& graph = block.dataflow();
    EXPECT_EQ(graph.nodes().size(), 3);
    EXPECT_EQ(graph.edges().size(), 2);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::DataParallelismAnalysis>();
    auto& dependencies = analysis.get(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 1);
    EXPECT_EQ(dependencies.at("A"), analysis::Parallelism::PARALLEL);
}

TEST(TestDataParallelism, Map_1D_Copy) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("B", desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
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
    auto& B = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", base_desc},
                                        {{"_in", base_desc}});
    builder.add_memlet(block, A, "void", tasklet, "_in", {indvar});
    builder.add_memlet(block, tasklet, "_out", B, "void", {indvar});

    auto& graph = block.dataflow();
    EXPECT_EQ(graph.nodes().size(), 3);
    EXPECT_EQ(graph.edges().size(), 2);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::DataParallelismAnalysis>();
    auto& dependencies = analysis.get(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 2);
    EXPECT_EQ(dependencies.at("A"), analysis::Parallelism::READONLY);
    EXPECT_EQ(dependencies.at("B"), analysis::Parallelism::PARALLEL);
}

TEST(TestDataParallelism, Reduction_1D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("b", base_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
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
    auto& b = builder.add_access(block, "b");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", base_desc},
                                        {{"_in", base_desc}});
    builder.add_memlet(block, A, "void", tasklet, "_in", {indvar});
    builder.add_memlet(block, tasklet, "_out", b, "void", {});

    auto& graph = block.dataflow();
    EXPECT_EQ(graph.nodes().size(), 3);
    EXPECT_EQ(graph.edges().size(), 2);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::DataParallelismAnalysis>();
    auto& dependencies = analysis.get(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 2);
    EXPECT_EQ(dependencies.at("A"), analysis::Parallelism::READONLY);
    EXPECT_EQ(dependencies.at("b"), analysis::Parallelism::REDUCTION);
}

TEST(TestDataParallelism, Sequential_Left_Shift_1D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
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
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", base_desc},
                                        {{"_in", base_desc}});
    builder.add_memlet(block, A1, "void", tasklet, "_in",
                       {symbolic::sub(indvar, symbolic::integer(1))});
    builder.add_memlet(block, tasklet, "_out", A2, "void", {indvar});

    auto& graph = block.dataflow();
    EXPECT_EQ(graph.nodes().size(), 3);
    EXPECT_EQ(graph.edges().size(), 2);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::DataParallelismAnalysis>();
    auto& dependencies = analysis.get(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 1);
    EXPECT_EQ(dependencies.at("A"), analysis::Parallelism::DEPENDENT);
}

TEST(TestDataParallelism, Sequential_Partial_Sum_1D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
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
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::add, {"_out", base_desc},
                                        {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block, A1, "void", tasklet, "_in1",
                       {symbolic::sub(indvar, symbolic::integer(1))});
    builder.add_memlet(block, A2, "void", tasklet, "_in2", {indvar});
    builder.add_memlet(block, tasklet, "_out", A3, "void", {indvar});

    auto& graph = block.dataflow();
    EXPECT_EQ(graph.nodes().size(), 4);
    EXPECT_EQ(graph.edges().size(), 3);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::DataParallelismAnalysis>();
    auto& dependencies = analysis.get(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 1);
    EXPECT_EQ(dependencies.at("A"), analysis::Parallelism::DEPENDENT);
}

TEST(TestDataParallelism, Stencil_1D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("B", desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
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
    auto& tasklet =
        builder.add_tasklet(block, data_flow::TaskletCode::fma, {"_out", base_desc},
                            {{"_in1", base_desc}, {"_in2", base_desc}, {"_in3", base_desc}});
    builder.add_memlet(block, A1, "void", tasklet, "_in1",
                       {symbolic::sub(indvar, symbolic::integer(1))});
    builder.add_memlet(block, A2, "void", tasklet, "_in2", {indvar});
    builder.add_memlet(block, A3, "void", tasklet, "_in3",
                       {symbolic::add(indvar, symbolic::integer(1))});
    builder.add_memlet(block, tasklet, "_out", B, "void", {indvar});

    auto& graph = block.dataflow();
    EXPECT_EQ(graph.nodes().size(), 5);
    EXPECT_EQ(graph.edges().size(), 4);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::DataParallelismAnalysis>();
    auto& dependencies = analysis.get(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 2);
    EXPECT_EQ(dependencies.at("A"), analysis::Parallelism::READONLY);
    EXPECT_EQ(dependencies.at("B"), analysis::Parallelism::PARALLEL);
}

TEST(TestDataParallelism, Gather_1D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Int64);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("B", desc, true);
    builder.add_container("C", desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
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
    auto& block = builder.add_block(body);

    // Define indirection
    auto& A = builder.add_access(block, "A");
    auto& b = builder.add_access(block, "b");
    auto& tasklet1 = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", base_desc},
                                         {{"_in", base_desc}});
    builder.add_memlet(block, A, "void", tasklet1, "_in", {indvar});
    builder.add_memlet(block, tasklet1, "_out", b, "void", {});

    auto& A1 = builder.add_access(block, "B");
    auto& A2 = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", base_desc},
                                        {{"_in", base_desc}});
    builder.add_memlet(block, A1, "void", tasklet, "_in", {symbolic::symbol("b")});
    builder.add_memlet(block, tasklet, "_out", A2, "void", {indvar});

    auto& graph = block.dataflow();
    EXPECT_EQ(graph.nodes().size(), 6);
    EXPECT_EQ(graph.edges().size(), 4);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::DataParallelismAnalysis>();
    auto& dependencies = analysis.get(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 4);
    EXPECT_EQ(dependencies.at("A"), analysis::Parallelism::READONLY);
    EXPECT_EQ(dependencies.at("b"), analysis::Parallelism::PRIVATE);
    EXPECT_EQ(dependencies.at("B"), analysis::Parallelism::READONLY);
    EXPECT_EQ(dependencies.at("C"), analysis::Parallelism::PARALLEL);
}

TEST(TestDataParallelism, Scatter_1D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Int64);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("B", desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
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
    auto& block = builder.add_block(body);

    // Define indirection
    auto& A = builder.add_access(block, "A");
    auto& b = builder.add_access(block, "b");
    auto& tasklet1 = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", base_desc},
                                         {{"_in", base_desc}});
    builder.add_memlet(block, A, "void", tasklet1, "_in", {indvar});
    builder.add_memlet(block, tasklet1, "_out", b, "void", {});

    auto& A1 = builder.add_access(block, "B");
    auto& A2 = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::add, {"_out", base_desc},
                                        {{"_in", base_desc}, {"1", base_desc}});
    builder.add_memlet(block, A1, "void", tasklet, "_in", {symbolic::symbol("b")});
    builder.add_memlet(block, tasklet, "_out", A2, "void", {symbolic::symbol("b")});

    auto& graph = block.dataflow();
    EXPECT_EQ(graph.nodes().size(), 6);
    EXPECT_EQ(graph.edges().size(), 4);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::DataParallelismAnalysis>();
    auto& dependencies = analysis.get(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 3);
    EXPECT_EQ(dependencies.at("A"), analysis::Parallelism::READONLY);
    EXPECT_EQ(dependencies.at("b"), analysis::Parallelism::PRIVATE);
    EXPECT_EQ(dependencies.at("B"), analysis::Parallelism::DEPENDENT);
}

TEST(TestDataParallelism, Map_2D_Copy) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Pointer desc2(*desc.clone());
    builder.add_container("A", desc2, true);
    builder.add_container("B", desc2, true);

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
    auto& B = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", base_desc},
                                        {{"_in", base_desc}});
    builder.add_memlet(block, A, "void", tasklet, "_in", {indvar1, indvar2});
    builder.add_memlet(block, tasklet, "_out", B, "void", {indvar1, indvar2});

    auto& graph = block.dataflow();
    EXPECT_EQ(graph.nodes().size(), 3);
    EXPECT_EQ(graph.edges().size(), 2);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::DataParallelismAnalysis>();
    auto& dependencies1 = analysis.get(loop1);
    auto& dependencies2 = analysis.get(loop2);

    // Check
    EXPECT_EQ(dependencies1.size(), 4);
    EXPECT_EQ(dependencies1.at("N"), analysis::Parallelism::READONLY);
    EXPECT_EQ(dependencies1.at("j"), analysis::Parallelism::PRIVATE);
    EXPECT_EQ(dependencies1.at("A"), analysis::Parallelism::READONLY);
    EXPECT_EQ(dependencies1.at("B"), analysis::Parallelism::PARALLEL);

    EXPECT_EQ(dependencies2.size(), 3);
    EXPECT_EQ(dependencies2.at("i"), analysis::Parallelism::READONLY);
    EXPECT_EQ(dependencies2.at("A"), analysis::Parallelism::READONLY);
    EXPECT_EQ(dependencies2.at("B"), analysis::Parallelism::PARALLEL);
}

TEST(TestDataParallelism, Map_2D_Transpose) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Pointer desc2(*desc.clone());
    builder.add_container("A", desc2, true);
    builder.add_container("B", desc2, true);

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
    auto& B = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", base_desc},
                                        {{"_in", base_desc}});
    builder.add_memlet(block, A, "void", tasklet, "_in", {indvar1, indvar2});
    builder.add_memlet(block, tasklet, "_out", B, "void", {indvar2, indvar1});

    auto& graph = block.dataflow();
    EXPECT_EQ(graph.nodes().size(), 3);
    EXPECT_EQ(graph.edges().size(), 2);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::DataParallelismAnalysis>();
    auto& dependencies1 = analysis.get(loop1);
    auto& dependencies2 = analysis.get(loop2);

    // Check
    EXPECT_EQ(dependencies1.size(), 4);
    EXPECT_EQ(dependencies1.at("N"), analysis::Parallelism::READONLY);
    EXPECT_EQ(dependencies1.at("j"), analysis::Parallelism::PRIVATE);
    EXPECT_EQ(dependencies1.at("A"), analysis::Parallelism::READONLY);
    EXPECT_EQ(dependencies1.at("B"), analysis::Parallelism::PARALLEL);

    EXPECT_EQ(dependencies2.size(), 3);
    EXPECT_EQ(dependencies2.at("i"), analysis::Parallelism::READONLY);
    EXPECT_EQ(dependencies2.at("A"), analysis::Parallelism::READONLY);
    EXPECT_EQ(dependencies2.at("B"), analysis::Parallelism::PARALLEL);
}

TEST(TestDataParallelism, Sequential_2D_Transpose) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Pointer desc2(*desc.clone());
    builder.add_container("A", desc2, true);

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
    auto& B = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", base_desc},
                                        {{"_in", base_desc}});
    builder.add_memlet(block, A, "void", tasklet, "_in", {indvar1, indvar2});
    builder.add_memlet(block, tasklet, "_out", B, "void", {indvar2, indvar1});

    auto& graph = block.dataflow();
    EXPECT_EQ(graph.nodes().size(), 3);
    EXPECT_EQ(graph.edges().size(), 2);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::DataParallelismAnalysis>();
    auto& dependencies1 = analysis.get(loop1);
    auto& dependencies2 = analysis.get(loop2);

    // Check
    EXPECT_EQ(dependencies1.size(), 3);
    EXPECT_EQ(dependencies1.at("N"), analysis::Parallelism::READONLY);
    EXPECT_EQ(dependencies1.at("j"), analysis::Parallelism::PRIVATE);
    EXPECT_EQ(dependencies1.at("A"), analysis::Parallelism::DEPENDENT);

    EXPECT_EQ(dependencies2.size(), 2);
    EXPECT_EQ(dependencies2.at("i"), analysis::Parallelism::READONLY);
    EXPECT_EQ(dependencies2.at("A"), analysis::Parallelism::PARALLEL);
}

TEST(TestDataParallelism, Reduction_2D_Inner) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Pointer desc2(*desc.clone());
    builder.add_container("A", desc2, true);
    builder.add_container("B", desc, true);

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
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::add, {"_out", base_desc},
                                        {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block, A, "void", tasklet, "_in1", {indvar1, indvar2});
    builder.add_memlet(block, B1, "void", tasklet, "_in2", {indvar1});
    builder.add_memlet(block, tasklet, "_out", B2, "void", {indvar1});

    auto& graph = block.dataflow();
    EXPECT_EQ(graph.nodes().size(), 4);
    EXPECT_EQ(graph.edges().size(), 3);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::DataParallelismAnalysis>();
    auto& dependencies1 = analysis.get(loop1);
    auto& dependencies2 = analysis.get(loop2);

    // Check
    EXPECT_EQ(dependencies1.size(), 4);
    EXPECT_EQ(dependencies1.at("N"), analysis::Parallelism::READONLY);
    EXPECT_EQ(dependencies1.at("j"), analysis::Parallelism::PRIVATE);
    EXPECT_EQ(dependencies1.at("A"), analysis::Parallelism::READONLY);
    EXPECT_EQ(dependencies1.at("B"), analysis::Parallelism::PARALLEL);

    EXPECT_EQ(dependencies2.size(), 3);
    EXPECT_EQ(dependencies2.at("i"), analysis::Parallelism::READONLY);
    EXPECT_EQ(dependencies2.at("A"), analysis::Parallelism::READONLY);
    EXPECT_EQ(dependencies2.at("B"), analysis::Parallelism::REDUCTION);
}

TEST(TestDataParallelism, Reduction_2D_Outer) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Pointer desc2(*desc.clone());
    builder.add_container("A", desc2, true);
    builder.add_container("B", desc, true);

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
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::add, {"_out", base_desc},
                                        {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block, A, "void", tasklet, "_in1", {indvar1, indvar2});
    builder.add_memlet(block, B1, "void", tasklet, "_in2", {indvar2});
    builder.add_memlet(block, tasklet, "_out", B2, "void", {indvar2});

    auto& graph = block.dataflow();
    EXPECT_EQ(graph.nodes().size(), 4);
    EXPECT_EQ(graph.edges().size(), 3);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::DataParallelismAnalysis>();
    auto& dependencies1 = analysis.get(loop1);
    auto& dependencies2 = analysis.get(loop2);

    // Check
    EXPECT_EQ(dependencies1.size(), 4);
    EXPECT_EQ(dependencies1.at("N"), analysis::Parallelism::READONLY);
    EXPECT_EQ(dependencies1.at("j"), analysis::Parallelism::PRIVATE);
    EXPECT_EQ(dependencies1.at("A"), analysis::Parallelism::READONLY);
    EXPECT_EQ(dependencies1.at("B"), analysis::Parallelism::DEPENDENT);

    EXPECT_EQ(dependencies2.size(), 3);
    EXPECT_EQ(dependencies2.at("i"), analysis::Parallelism::READONLY);
    EXPECT_EQ(dependencies2.at("A"), analysis::Parallelism::READONLY);
    EXPECT_EQ(dependencies2.at("B"), analysis::Parallelism::PARALLEL);
}

TEST(TestDataParallelism, Intervals_Disjoint_1D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("N", sym_desc, true);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);

    // Define loop
    auto indvar1 = symbolic::symbol("i");
    auto init1 = symbolic::integer(0);
    auto condition1 = symbolic::Lt(indvar1, symbolic::symbol("N"));
    auto update1 = symbolic::add(indvar1, symbolic::integer(1));

    auto& loop1 = builder.add_for(root, indvar1, condition1, init1, update1);
    auto& body1 = loop1.root();

    // Add computation
    auto& block = builder.add_block(body1);
    auto& A = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", base_desc},
                                        {{"_in", base_desc}});
    builder.add_memlet(block, A, "void", tasklet, "_in", {indvar1});
    builder.add_memlet(block, tasklet, "_out", B, "void",
                       {symbolic::add(indvar1, symbolic::symbol("N"))});

    auto& graph = block.dataflow();
    EXPECT_EQ(graph.nodes().size(), 3);
    EXPECT_EQ(graph.edges().size(), 2);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::DataParallelismAnalysis>();
    auto& dependencies1 = analysis.get(loop1);

    // Check
    EXPECT_EQ(dependencies1.size(), 2);
    EXPECT_EQ(dependencies1.at("N"), analysis::Parallelism::READONLY);
    EXPECT_EQ(dependencies1.at("A"), analysis::Parallelism::PARALLEL);
}

TEST(TestDataParallelism, Triangle_2D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Pointer desc2(*desc.clone());
    builder.add_container("A", desc2, true);

    // Define loop
    auto indvar1 = symbolic::symbol("i");
    auto init1 = symbolic::integer(0);
    auto condition1 =
        symbolic::Lt(indvar1, symbolic::sub(symbolic::symbol("N"), symbolic::integer(1)));
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
    auto& A = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", base_desc},
                                        {{"_in", base_desc}});
    builder.add_memlet(block, A, "void", tasklet, "_in", {indvar1, indvar2});
    builder.add_memlet(block, tasklet, "_out", B, "void", {indvar2, indvar1});

    auto& graph = block.dataflow();
    EXPECT_EQ(graph.nodes().size(), 3);
    EXPECT_EQ(graph.edges().size(), 2);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::DataParallelismAnalysis>();
    auto& dependencies1 = analysis.get(loop1);
    auto& dependencies2 = analysis.get(loop2);

    // Check
    EXPECT_EQ(dependencies1.size(), 3);
    EXPECT_EQ(dependencies1.at("N"), analysis::Parallelism::READONLY);
    EXPECT_EQ(dependencies1.at("j"), analysis::Parallelism::PRIVATE);
    EXPECT_EQ(dependencies1.at("A"), analysis::Parallelism::PARALLEL);

    EXPECT_EQ(dependencies2.size(), 2);
    EXPECT_EQ(dependencies2.at("i"), analysis::Parallelism::READONLY);
    EXPECT_EQ(dependencies2.at("A"), analysis::Parallelism::PARALLEL);
}

TEST(TestDataParallelism, Triangle_2D_2) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Pointer desc2(*desc.clone());
    builder.add_container("A", desc2, true);

    // Define loop
    auto indvar1 = symbolic::symbol("i");
    auto init1 = symbolic::integer(0);
    auto condition1 =
        symbolic::Lt(indvar1, symbolic::sub(symbolic::symbol("N"), symbolic::integer(1)));
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
    auto& A = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", base_desc},
                                        {{"_in", base_desc}});
    builder.add_memlet(block, A, "void", tasklet, "_in", {indvar1, indvar2});
    builder.add_memlet(block, tasklet, "_out", B, "void", {indvar2, indvar1});

    auto& graph = block.dataflow();
    EXPECT_EQ(graph.nodes().size(), 3);
    EXPECT_EQ(graph.edges().size(), 2);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::DataParallelismAnalysis>();
    auto& dependencies1 = analysis.get(loop1);
    auto& dependencies2 = analysis.get(loop2);

    // Check
    EXPECT_EQ(dependencies1.size(), 3);
    EXPECT_EQ(dependencies1.at("N"), analysis::Parallelism::READONLY);
    EXPECT_EQ(dependencies1.at("j"), analysis::Parallelism::PRIVATE);
    EXPECT_EQ(dependencies1.at("A"), analysis::Parallelism::PARALLEL);

    EXPECT_EQ(dependencies2.size(), 2);
    EXPECT_EQ(dependencies2.at("i"), analysis::Parallelism::READONLY);
    EXPECT_EQ(dependencies2.at("A"), analysis::Parallelism::PARALLEL);
}

TEST(TestDataParallelism, Temporal_Loop_1D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("T", sym_desc, true);
    builder.add_container("t", sym_desc);
    builder.add_container("i_1", sym_desc);
    builder.add_container("i_2", sym_desc);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("B", desc, true);

    auto indvar1 = symbolic::symbol("t");
    auto init1 = symbolic::integer(0);
    auto condition1 =
        symbolic::Lt(indvar1, symbolic::sub(symbolic::symbol("T"), symbolic::integer(1)));
    auto update1 = symbolic::add(indvar1, symbolic::integer(1));
    auto& loop1 = builder.add_for(root, indvar1, condition1, init1, update1);
    auto& body1 = loop1.root();

    auto indvar2 = symbolic::symbol("i_1");
    auto init2 = symbolic::integer(0);
    auto condition2 = symbolic::Lt(indvar2, symbolic::symbol("N"));
    auto update2 = symbolic::add(indvar2, symbolic::integer(1));
    auto& loop2 = builder.add_for(body1, indvar2, condition2, init2, update2);
    auto& body2 = loop2.root();
    {
        auto& block = builder.add_block(body2);
        auto& input_node = builder.add_access(block, "A");
        auto& output_node = builder.add_access(block, "B");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::add, {"_out", base_desc},
                                            {{"_in", base_desc}, {"1", base_desc}});
        builder.add_memlet(block, input_node, "void", tasklet, "_in", {indvar2});
        builder.add_memlet(block, tasklet, "_out", output_node, "void", {indvar2});
    }

    auto indvar3 = symbolic::symbol("i_2");
    auto init3 = symbolic::integer(0);
    auto condition3 = symbolic::Lt(indvar3, symbolic::symbol("N"));
    auto update3 = symbolic::add(indvar3, symbolic::integer(1));
    auto& loop3 = builder.add_for(body1, indvar3, condition3, init3, update3);
    auto& body3 = loop3.root();
    {
        auto& block = builder.add_block(body3);
        auto& input_node = builder.add_access(block, "B");
        auto& output_node = builder.add_access(block, "A");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                            {"_out", base_desc}, {{"_in", base_desc}});
        builder.add_memlet(block, input_node, "void", tasklet, "_in", {indvar3});
        builder.add_memlet(block, tasklet, "_out", output_node, "void", {indvar3});
    }

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::DataParallelismAnalysis>();
    auto& dependencies1 = analysis.get(loop1);
    auto& dependencies2 = analysis.get(loop2);
    auto& dependencies3 = analysis.get(loop3);

    // Check
    EXPECT_EQ(dependencies1.size(), 5);
    EXPECT_EQ(dependencies1.at("N"), analysis::Parallelism::READONLY);
    EXPECT_EQ(dependencies1.at("i_1"), analysis::Parallelism::PRIVATE);
    EXPECT_EQ(dependencies1.at("i_2"), analysis::Parallelism::PRIVATE);
    EXPECT_EQ(dependencies1.at("A"), analysis::Parallelism::DEPENDENT);
    EXPECT_EQ(dependencies1.at("B"), analysis::Parallelism::DEPENDENT);

    EXPECT_EQ(dependencies2.size(), 2);
    EXPECT_EQ(dependencies2.at("A"), analysis::Parallelism::READONLY);
    EXPECT_EQ(dependencies2.at("B"), analysis::Parallelism::PARALLEL);

    EXPECT_EQ(dependencies3.size(), 2);
    EXPECT_EQ(dependencies3.at("B"), analysis::Parallelism::READONLY);
    EXPECT_EQ(dependencies3.at("A"), analysis::Parallelism::PARALLEL);
}

TEST(TestDataParallelism, Conditional_Tasklets_Readonly) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc, true);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
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

    // Add computation
    auto& block2 = builder.add_block(body1);
    auto& A_in = builder.add_access(block2, "A");
    auto& A_out = builder.add_access(block2, "A");
    auto& tasklet = builder.add_tasklet(block2, data_flow::TaskletCode::assign, {"_out", base_desc},
                                        {{"_in", base_desc}});
    builder.add_memlet(block2, A_in, "void", tasklet, "_in", {indvar1});
    builder.add_memlet(block2, tasklet, "_out", A_out, "void", {indvar1});
    tasklet.condition() = symbolic::Gt(symbolic::symbol("j"), symbolic::integer(0));

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::DataParallelismAnalysis>();
    auto& dependencies1 = analysis.get(loop1);
    EXPECT_EQ(dependencies1.size(), 2);
    EXPECT_EQ(dependencies1.at("j"), analysis::Parallelism::READONLY);
    EXPECT_EQ(dependencies1.at("A"), analysis::Parallelism::PARALLEL);
}

TEST(TestDataParallelism, Conditional_Tasklets_Private) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
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

    // Add computation
    auto& block1 = builder.add_block(body1);
    auto& B = builder.add_access(block1, "B");
    auto& j = builder.add_access(block1, "j");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign,
                                         {"_out", base_desc}, {{"_in", base_desc}});
    builder.add_memlet(block1, B, "void", tasklet1, "_in", {indvar1});
    builder.add_memlet(block1, tasklet1, "_out", j, "void", {});

    auto& block2 = builder.add_block(body1);
    auto& A_in = builder.add_access(block2, "A");
    auto& A_out = builder.add_access(block2, "A");
    auto& tasklet = builder.add_tasklet(block2, data_flow::TaskletCode::assign, {"_out", base_desc},
                                        {{"_in", base_desc}});
    builder.add_memlet(block2, A_in, "void", tasklet, "_in", {indvar1});
    builder.add_memlet(block2, tasklet, "_out", A_out, "void", {indvar1});
    tasklet.condition() = symbolic::Gt(symbolic::symbol("j"), symbolic::integer(0));

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::DataParallelismAnalysis>();
    auto& dependencies1 = analysis.get(loop1);
    EXPECT_EQ(dependencies1.size(), 3);
    EXPECT_EQ(dependencies1.at("j"), analysis::Parallelism::PRIVATE);
    EXPECT_EQ(dependencies1.at("A"), analysis::Parallelism::PARALLEL);
    EXPECT_EQ(dependencies1.at("B"), analysis::Parallelism::READONLY);
}

TEST(TestDataParallelism, Conditional_Tasklets_Dependent) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);

    auto& init_block = builder.add_block(root, {{symbolic::symbol("j"), symbolic::integer(0)}});

    // Define loop
    auto bound1 = symbolic::symbol("N");
    auto indvar1 = symbolic::symbol("i");
    auto init1 = symbolic::integer(0);
    auto condition1 = symbolic::Lt(indvar1, bound1);
    auto update1 = symbolic::add(indvar1, symbolic::integer(1));

    auto& loop1 = builder.add_for(root, indvar1, condition1, init1, update1);
    auto& body1 = loop1.root();

    // Add computation
    auto& block1 = builder.add_block(body1);
    auto& j_in = builder.add_access(block1, "j");
    auto& j_out = builder.add_access(block1, "j");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::add, {"_out", base_desc},
                                         {{"_in", base_desc}, {"1", base_desc}});
    builder.add_memlet(block1, j_in, "void", tasklet1, "_in", {});
    builder.add_memlet(block1, tasklet1, "_out", j_out, "void", {});

    auto& block2 = builder.add_block(body1);
    auto& A_in = builder.add_access(block2, "A");
    auto& A_out = builder.add_access(block2, "A");
    auto& tasklet = builder.add_tasklet(block2, data_flow::TaskletCode::assign, {"_out", base_desc},
                                        {{"_in", base_desc}});
    builder.add_memlet(block2, A_in, "void", tasklet, "_in", {indvar1});
    builder.add_memlet(block2, tasklet, "_out", A_out, "void", {indvar1});
    tasklet.condition() = symbolic::Gt(symbolic::symbol("j"), symbolic::integer(0));

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::DataParallelismAnalysis>();
    auto& dependencies1 = analysis.get(loop1);
    EXPECT_EQ(dependencies1.size(), 0);
}

TEST(TestDataParallelism, AffineParameters) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("a", sym_desc, true);
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
    auto& A = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", base_desc},
                                        {{"0", base_desc}});
    builder.add_memlet(
        block, tasklet, "_out", A, "void",
        {symbolic::add(indvar, symbolic::mul(symbolic::symbol("b"), symbolic::symbol("a")))});

    auto& graph = block.dataflow();
    EXPECT_EQ(graph.nodes().size(), 2);
    EXPECT_EQ(graph.edges().size(), 1);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::DataParallelismAnalysis>();
    auto& dependencies = analysis.get(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 3);
    EXPECT_EQ(dependencies.at("A"), analysis::Parallelism::PARALLEL);
    EXPECT_EQ(dependencies.at("a"), analysis::Parallelism::READONLY);
    EXPECT_EQ(dependencies.at("b"), analysis::Parallelism::READONLY);
}

TEST(TestDataParallelism, Map_2D_Linearized) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);

    types::Scalar sym_desc(types::PrimitiveType::Int32);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    auto& loop_2 = builder.add_for(
        body, symbolic::symbol("j"), symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("M")),
        symbolic::integer(0), symbolic::add(symbolic::symbol("j"), symbolic::integer(1)));
    auto& body_2 = loop_2.root();

    // Add computation
    auto& block = builder.add_block(body_2);
    auto& A = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", base_desc},
                                        {{"0", base_desc}});
    builder.add_memlet(
        block, tasklet, "_out", A, "void",
        {symbolic::add(symbolic::symbol("j"),
                       symbolic::mul(symbolic::symbol("i"), symbolic::symbol("M")))});

    auto& graph = block.dataflow();
    EXPECT_EQ(graph.nodes().size(), 2);
    EXPECT_EQ(graph.edges().size(), 1);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::DataParallelismAnalysis>();

    // Check
    auto& dependencies = analysis.get(loop);
    EXPECT_EQ(dependencies.size(), 3);
    EXPECT_EQ(dependencies.at("A"), analysis::Parallelism::PARALLEL);
    EXPECT_EQ(dependencies.at("j"), analysis::Parallelism::PRIVATE);
    EXPECT_EQ(dependencies.at("M"), analysis::Parallelism::READONLY);

    auto& dependencies_2 = analysis.get(loop_2);
    EXPECT_EQ(dependencies_2.size(), 3);
    EXPECT_EQ(dependencies_2.at("i"), analysis::Parallelism::READONLY);
    EXPECT_EQ(dependencies_2.at("A"), analysis::Parallelism::PARALLEL);
    EXPECT_EQ(dependencies_2.at("M"), analysis::Parallelism::READONLY);
}

TEST(TestDataParallelism, KernelTestBasic) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_NV_GLOBAL);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("B", desc, true);

    types::Scalar sym_desc(types::PrimitiveType::Int32);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

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
    auto& B = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", base_desc},
                                        {{"_in", base_desc}});
    builder.add_memlet(
        block, A, "void", tasklet, "_in",
        {symbolic::add(symbolic::threadIdx_x(),
                       symbolic::mul(symbolic::blockDim_x(), symbolic::symbol("blockIdx.x")))});
    builder.add_memlet(
        block, tasklet, "_out", B, "void",
        {symbolic::add(symbolic::threadIdx_x(),
                       symbolic::mul(symbolic::blockDim_x(), symbolic::symbol("blockIdx.x")))});

    auto& graph = block.dataflow();
    EXPECT_EQ(graph.nodes().size(), 3);
    EXPECT_EQ(graph.edges().size(), 2);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::DataParallelismAnalysis>();

    // Check
    auto& dependencies = analysis.get(loop);
    EXPECT_EQ(dependencies.size(), 5);
    EXPECT_EQ(dependencies.at("B"), analysis::Parallelism::REDUCTION);
    EXPECT_EQ(dependencies.at("A"), analysis::Parallelism::READONLY);
}

TEST(TestDataParallelism, KernelTest) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_NV_GLOBAL);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("B", desc, true);

    types::Scalar sym_desc(types::PrimitiveType::Int32);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

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
    auto& B = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", base_desc},
                                        {{"_in", base_desc}});
    builder.add_memlet(
        block, A, "void", tasklet, "_in",
        {symbolic::add(
            indvar, symbolic::mul(symbolic::integer(512),
                                  symbolic::add(symbolic::threadIdx_x(),
                                                symbolic::mul(symbolic::blockDim_x(),
                                                              symbolic::symbol("blockIdx.x")))))});
    builder.add_memlet(
        block, tasklet, "_out", B, "void",
        {symbolic::add(
            indvar, symbolic::mul(symbolic::integer(512),
                                  symbolic::add(symbolic::threadIdx_x(),
                                                symbolic::mul(symbolic::blockDim_x(),
                                                              symbolic::symbol("blockIdx.x")))))});

    auto& graph = block.dataflow();
    EXPECT_EQ(graph.nodes().size(), 3);
    EXPECT_EQ(graph.edges().size(), 2);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::DataParallelismAnalysis>();

    // Check
    auto& dependencies = analysis.get(loop);
    EXPECT_EQ(dependencies.size(), 5);
    EXPECT_EQ(dependencies.at("B"), analysis::Parallelism::PARALLEL);
    EXPECT_EQ(dependencies.at("A"), analysis::Parallelism::READONLY);
}

TEST(TestDataParallelism, KernelTestMult) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_NV_GLOBAL);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("B", desc, true);

    types::Scalar sym_desc(types::PrimitiveType::Int32);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

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
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", base_desc},
                                        {{"_in", base_desc}});
    builder.add_memlet(
        block, A, "void", tasklet, "_in",
        {symbolic::add(
            symbolic::mul(symbolic::integer(512), indvar),
            symbolic::add(symbolic::threadIdx_x(),
                          symbolic::mul(symbolic::blockDim_x(), symbolic::symbol("blockIdx.x"))))});

    auto& B = builder.add_access(block, "B");
    builder.add_memlet(
        block, tasklet, "_out", B, "void",
        {symbolic::add(
            symbolic::mul(symbolic::integer(512), indvar),
            symbolic::add(symbolic::threadIdx_x(),
                          symbolic::mul(symbolic::blockDim_x(), symbolic::symbol("blockIdx.x"))))});

    auto& graph = block.dataflow();
    EXPECT_EQ(graph.nodes().size(), 3);
    EXPECT_EQ(graph.edges().size(), 2);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::DataParallelismAnalysis>();

    // Check
    auto& dependencies = analysis.get(loop);
    EXPECT_EQ(dependencies.size(), 5);
    EXPECT_EQ(dependencies.at("B"), analysis::Parallelism::PARALLEL);
    EXPECT_EQ(dependencies.at("A"), analysis::Parallelism::READONLY);
}

TEST(TestDataParallelism, KernelTestTiled) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_NV_GLOBAL);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("B", desc, true);

    types::Array array_desc(base_desc, {symbolic::integer(8)});
    types::Array array_desc2(array_desc, {symbolic::integer(512)});
    builder.add_container("B_shared", array_desc2);

    types::Scalar sym_desc(types::PrimitiveType::UInt32);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("i_shared", sym_desc);
    builder.add_container("i_access", sym_desc);

    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(8));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    auto indvar_shared = symbolic::symbol("i_shared");
    auto shared_bound =
        symbolic::And(symbolic::Lt(indvar_shared, symbolic::add(symbolic::integer(8), indvar)),
                      symbolic::Lt(indvar_shared, symbolic::max(symbolic::integer(0), bound)));
    auto shared_init = indvar;
    auto shared_update = symbolic::add(indvar_shared, symbolic::integer(1));

    auto& loop_shared =
        builder.add_for(body, indvar_shared, shared_bound, shared_init, shared_update);
    auto& body_shared = loop_shared.root();

    auto& block_shared = builder.add_block(body_shared);
    auto& B = builder.add_access(block_shared, "B");
    auto& B_shared = builder.add_access(block_shared, "B_shared");

    auto& tasklet_shared = builder.add_tasklet(block_shared, data_flow::TaskletCode::assign,
                                               {"_out", base_desc}, {{"_in", base_desc}});

    auto& shared_in = builder.add_memlet(
        block_shared, B, "void", tasklet_shared, "_in",
        {symbolic::add(
            symbolic::mul(symbolic::integer(512), indvar_shared),
            symbolic::add(symbolic::threadIdx_x(),
                          symbolic::mul(symbolic::blockDim_x(), symbolic::symbol("blockIdx.x"))))});

    auto& shared_out =
        builder.add_memlet(block_shared, tasklet_shared, "_out", B_shared, "void",
                           {symbolic::threadIdx_x(), symbolic::sub(indvar_shared, indvar)});

    auto& sync_block = builder.add_block(body);
    auto& libnode =
        builder.add_library_node(sync_block, data_flow::LibraryNodeCode::barrier_local, {}, {});

    auto indvar_access = symbolic::symbol("i_access");
    auto access_bound =
        symbolic::And(symbolic::Lt(indvar_access, symbolic::add(symbolic::integer(8), indvar)),
                      symbolic::Lt(indvar_access, symbolic::max(symbolic::integer(0), bound)));
    auto access_init = indvar;
    auto access_update = symbolic::add(indvar_access, symbolic::integer(1));

    auto& loop_access =
        builder.add_for(body, indvar_access, access_bound, access_init, access_update);
    auto& body_access = loop_access.root();

    auto& block_access = builder.add_block(body_access);
    auto& B_shared_access = builder.add_access(block_access, "B_shared");
    auto& A = builder.add_access(block_access, "A");

    auto& tasklet_access = builder.add_tasklet(block_access, data_flow::TaskletCode::assign,
                                               {"_out", base_desc}, {{"_in", base_desc}});

    auto& access_in =
        builder.add_memlet(block_access, B_shared_access, "void", tasklet_access, "_in",
                           {symbolic::threadIdx_x(), symbolic::sub(indvar_access, indvar)});

    auto& access_out = builder.add_memlet(
        block_access, tasklet_access, "_out", A, "void",
        {symbolic::add(
            symbolic::mul(symbolic::integer(512), indvar_access),
            symbolic::add(symbolic::threadIdx_x(),
                          symbolic::mul(symbolic::blockDim_x(), symbolic::symbol("blockIdx.x"))))});

    auto& graph_shared = block_shared.dataflow();
    EXPECT_EQ(graph_shared.nodes().size(), 3);
    EXPECT_EQ(graph_shared.edges().size(), 2);

    auto& graph_access = block_access.dataflow();
    EXPECT_EQ(graph_access.nodes().size(), 3);
    EXPECT_EQ(graph_access.edges().size(), 2);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::DataParallelismAnalysis>();
    auto& assumptions = analysis_manager.get<analysis::AssumptionsAnalysis>();

    // Check
    auto& dependencies_shared = analysis.get(loop_shared);
    EXPECT_EQ(dependencies_shared.size(), 6);
    EXPECT_EQ(dependencies_shared.at("B"), analysis::Parallelism::READONLY);
    EXPECT_EQ(dependencies_shared.at("B_shared"), analysis::Parallelism::PARALLEL);

    EXPECT_TRUE(symbolic::eq(assumptions.get(root).at(indvar_shared).lower_bound(), indvar));
    EXPECT_TRUE(symbolic::eq(assumptions.get(root).at(indvar_shared).upper_bound(),
                             symbolic::min(symbolic::add(indvar, symbolic::integer(7)),
                                           symbolic::sub(symbolic::max(symbolic::integer(0), bound),
                                                         symbolic::integer(1)))));

    auto& dependencies_access = analysis.get(loop_access);
    EXPECT_EQ(dependencies_access.at("A"), analysis::Parallelism::PARALLEL);
    EXPECT_EQ(dependencies_access.at("B_shared"), analysis::Parallelism::READONLY);
}

TEST(TestDataParallelism, Rodinia_SRAD) {
    /**
            for (int i = 0 ; i < rows ; i++) {
                for (int j = 0; j < cols; j++) {
                    Jc = J[i * cols + j];

                    // directional derivates
                    dN[i * cols + j] = J[iN[i] * cols + j] - Jc;
                    dS[i * cols + j] = J[iS[i] * cols + j] - Jc;
                    dW[i * cols + j] = J[i * cols + jW[j]] - Jc;
                    dE[i * cols + j] = J[i * cols + jE[j]] - Jc;

                    G2 = (dN[i * cols + j]*dN[i * cols + j] + dS[i * cols + j]*dS[i * cols + j]
                        + dW[i * cols + j]*dW[i * cols + j] + dE[i * cols + j]*dE[i * cols + j]) /
       Jc;

                    L = (dN[i * cols + j] + dS[i * cols + j] + dW[i * cols + j] + dE[i * cols + j])
       / Jc;

                    // diffusion coefficent (equ 33)
                    c[i * cols + j] = L - q0sqr;

                    // saturate diffusion coefficent
                    if (c[i * cols + j] < 0) {c[i * cols + j] = 0;}
                    else if (c[i * cols + j] > 1) {c[i * cols + j] = 1;}
                }
            }
        */

    builder::StructuredSDFGBuilder builder("srad", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::Int32);
    builder.add_container("rows", sym_desc, true);
    builder.add_container("cols", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    builder.add_container("iN_i", sym_desc);
    builder.add_container("iS_i", sym_desc);
    builder.add_container("jW_j", sym_desc);
    builder.add_container("jE_j", sym_desc);

    types::Scalar element_desc(types::PrimitiveType::Float);
    builder.add_container("G2", element_desc);
    builder.add_container("L", element_desc);
    builder.add_container("Jc", element_desc);
    builder.add_container("q0sqr", element_desc, true);

    types::Pointer ptr_desc(element_desc);
    builder.add_container("J", ptr_desc, true);
    builder.add_container("dN", ptr_desc, true);
    builder.add_container("dS", ptr_desc, true);
    builder.add_container("dW", ptr_desc, true);
    builder.add_container("dE", ptr_desc, true);
    builder.add_container("c", ptr_desc, true);

    types::Pointer ptr_sym(sym_desc);
    builder.add_container("iN", ptr_sym, true);
    builder.add_container("iS", ptr_sym, true);
    builder.add_container("jW", ptr_sym, true);
    builder.add_container("jE", ptr_sym, true);

    auto& root = builder.subject().root();

    auto& loop_i = builder.add_for(
        root, symbolic::symbol("i"), symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("rows")),
        symbolic::integer(0), symbolic::add(symbolic::symbol("i"), symbolic::integer(1)));
    auto& body_loop_i = loop_i.root();

    auto& loop_j = builder.add_for(body_loop_i, symbolic::symbol("j"),
                                   symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("cols")),
                                   symbolic::integer(0),
                                   symbolic::add(symbolic::symbol("j"), symbolic::integer(1)));
    auto& body_loop_j = loop_j.root();

    {
        auto& block1 = builder.add_block(body_loop_j);
        auto& iN_i_node = builder.add_access(block1, "iN_i");
        auto& iN_node = builder.add_access(block1, "iN");
        auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign,
                                             {"_out", sym_desc}, {{"_in", sym_desc}});
        builder.add_memlet(block1, iN_node, "void", tasklet1, "_in", {symbolic::symbol("i")});
        builder.add_memlet(block1, tasklet1, "_out", iN_i_node, "void", {});

        auto& block2 = builder.add_block(body_loop_j);
        auto& iS_i_node = builder.add_access(block2, "iS_i");
        auto& iS_node = builder.add_access(block2, "iS");
        auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign,
                                             {"_out", sym_desc}, {{"_in", sym_desc}});
        builder.add_memlet(block2, iS_node, "void", tasklet2, "_in", {symbolic::symbol("i")});
        builder.add_memlet(block2, tasklet2, "_out", iS_i_node, "void", {});

        auto& block3 = builder.add_block(body_loop_j);
        auto& jW_j_node = builder.add_access(block3, "jW_j");
        auto& jW_node = builder.add_access(block3, "jW");
        auto& tasklet3 = builder.add_tasklet(block3, data_flow::TaskletCode::assign,
                                             {"_out", sym_desc}, {{"_in", sym_desc}});
        builder.add_memlet(block3, jW_node, "void", tasklet3, "_in", {symbolic::symbol("j")});
        builder.add_memlet(block3, tasklet3, "_out", jW_j_node, "void", {});

        auto& block4 = builder.add_block(body_loop_j);
        auto& jE_j_node = builder.add_access(block4, "jE_j");
        auto& jE_node = builder.add_access(block4, "jE");
        auto& tasklet4 = builder.add_tasklet(block4, data_flow::TaskletCode::assign,
                                             {"_out", sym_desc}, {{"_in", sym_desc}});
        builder.add_memlet(block4, jE_node, "void", tasklet4, "_in", {symbolic::symbol("j")});
        builder.add_memlet(block4, tasklet4, "_out", jE_j_node, "void", {});
    }

    {
        auto& block = builder.add_block(body_loop_j);
        auto& Jc = builder.add_access(block, "Jc");
        auto& J = builder.add_access(block, "J");
        auto& tasklet11 = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                              {"_out", element_desc}, {{"_in", element_desc}});
        builder.add_memlet(
            block, J, "void", tasklet11, "_in",
            {symbolic::add(symbolic::mul(symbolic::symbol("i"), symbolic::symbol("cols")),
                           symbolic::symbol("j"))});
        builder.add_memlet(block, tasklet11, "_out", Jc, "void", {});
    }

    // directional derivates
    {
        auto& block1 = builder.add_block(body_loop_j);
        auto& dN = builder.add_access(block1, "dN");
        auto& J_node1 = builder.add_access(block1, "J");
        auto& Jc_node1 = builder.add_access(block1, "Jc");
        auto& tasklet1 =
            builder.add_tasklet(block1, data_flow::TaskletCode::sub, {"_out", element_desc},
                                {{"_in1", element_desc}, {"_in2", element_desc}});
        builder.add_memlet(
            block1, J_node1, "void", tasklet1, "_in1",
            {symbolic::add(symbolic::mul(symbolic::symbol("iN_i"), symbolic::symbol("cols")),
                           symbolic::symbol("j"))});
        builder.add_memlet(block1, Jc_node1, "void", tasklet1, "_in2", {});
        builder.add_memlet(
            block1, tasklet1, "_out", dN, "void",
            {symbolic::add(symbolic::mul(symbolic::symbol("i"), symbolic::symbol("cols")),
                           symbolic::symbol("j"))});

        auto& block2 = builder.add_block(body_loop_j);
        auto& dS = builder.add_access(block2, "dS");
        auto& J_node2 = builder.add_access(block2, "J");
        auto& Jc_node2 = builder.add_access(block2, "Jc");
        auto& tasklet2 =
            builder.add_tasklet(block2, data_flow::TaskletCode::sub, {"_out", element_desc},
                                {{"_in1", element_desc}, {"_in2", element_desc}});
        builder.add_memlet(
            block2, J_node2, "void", tasklet2, "_in1",
            {symbolic::add(symbolic::mul(symbolic::symbol("iS_i"), symbolic::symbol("cols")),
                           symbolic::symbol("j"))});
        builder.add_memlet(block2, Jc_node2, "void", tasklet2, "_in2", {});
        builder.add_memlet(
            block2, tasklet2, "_out", dS, "void",
            {symbolic::add(symbolic::mul(symbolic::symbol("i"), symbolic::symbol("cols")),
                           symbolic::symbol("j"))});

        auto& block3 = builder.add_block(body_loop_j);
        auto& dW = builder.add_access(block3, "dW");
        auto& J_node3 = builder.add_access(block3, "J");
        auto& Jc_node3 = builder.add_access(block3, "Jc");
        auto& tasklet3 =
            builder.add_tasklet(block3, data_flow::TaskletCode::sub, {"_out", element_desc},
                                {{"_in1", element_desc}, {"_in2", element_desc}});
        builder.add_memlet(
            block3, J_node3, "void", tasklet3, "_in1",
            {symbolic::add(symbolic::mul(symbolic::symbol("i"), symbolic::symbol("cols")),
                           symbolic::symbol("jW_j"))});
        builder.add_memlet(block3, Jc_node3, "void", tasklet3, "_in2", {});
        builder.add_memlet(
            block3, tasklet3, "_out", dW, "void",
            {symbolic::add(symbolic::mul(symbolic::symbol("i"), symbolic::symbol("cols")),
                           symbolic::symbol("j"))});

        auto& block4 = builder.add_block(body_loop_j);
        auto& dE = builder.add_access(block4, "dE");
        auto& J_node4 = builder.add_access(block4, "J");
        auto& Jc_node4 = builder.add_access(block4, "Jc");
        auto& tasklet4 =
            builder.add_tasklet(block4, data_flow::TaskletCode::sub, {"_out", element_desc},
                                {{"_in1", element_desc}, {"_in2", element_desc}});
        builder.add_memlet(
            block4, J_node4, "void", tasklet4, "_in1",
            {symbolic::add(symbolic::mul(symbolic::symbol("i"), symbolic::symbol("cols")),
                           symbolic::symbol("jE_j"))});
        builder.add_memlet(block4, Jc_node4, "void", tasklet4, "_in2", {});
        builder.add_memlet(
            block4, tasklet4, "_out", dE, "void",
            {symbolic::add(symbolic::mul(symbolic::symbol("i"), symbolic::symbol("cols")),
                           symbolic::symbol("j"))});
    }

    // G2
    {
        auto& block1 = builder.add_block(body_loop_j);
        auto& G2_node1 = builder.add_access(block1, "G2");
        auto& dN_node1 = builder.add_access(block1, "dN");
        auto& tasklet2 =
            builder.add_tasklet(block1, data_flow::TaskletCode::mul, {"_out", element_desc},
                                {{"_in1", element_desc}, {"_in2", element_desc}});
        builder.add_memlet(
            block1, dN_node1, "void", tasklet2, "_in1",
            {symbolic::add(symbolic::mul(symbolic::symbol("i"), symbolic::symbol("cols")),
                           symbolic::symbol("j"))});
        builder.add_memlet(
            block1, dN_node1, "void", tasklet2, "_in2",
            {symbolic::add(symbolic::mul(symbolic::symbol("i"), symbolic::symbol("cols")),
                           symbolic::symbol("j"))});
        builder.add_memlet(block1, tasklet2, "_out", G2_node1, "void", {});

        auto& block2 = builder.add_block(body_loop_j);
        auto& G2_node2 = builder.add_access(block2, "G2");
        auto& G2_node2_out = builder.add_access(block2, "G2");
        auto& dS_node2 = builder.add_access(block2, "dS");
        auto& tasklet3 = builder.add_tasklet(
            block2, data_flow::TaskletCode::fma, {"_out", element_desc},
            {{"_in1", element_desc}, {"_in2", element_desc}, {"_in3", element_desc}});
        builder.add_memlet(
            block2, dS_node2, "void", tasklet3, "_in1",
            {symbolic::add(symbolic::mul(symbolic::symbol("i"), symbolic::symbol("cols")),
                           symbolic::symbol("j"))});
        builder.add_memlet(
            block2, dS_node2, "void", tasklet3, "_in2",
            {symbolic::add(symbolic::mul(symbolic::symbol("i"), symbolic::symbol("cols")),
                           symbolic::symbol("j"))});
        builder.add_memlet(block2, G2_node2, "void", tasklet3, "_in3", {});
        builder.add_memlet(block2, tasklet3, "_out", G2_node2_out, "void", {});

        auto& block3 = builder.add_block(body_loop_j);
        auto& G2_node3 = builder.add_access(block3, "G2");
        auto& G2_node3_out = builder.add_access(block3, "G2");
        auto& dW_node3 = builder.add_access(block3, "dW");
        auto& tasklet4 = builder.add_tasklet(
            block3, data_flow::TaskletCode::fma, {"_out", element_desc},
            {{"_in1", element_desc}, {"_in2", element_desc}, {"_in3", element_desc}});
        builder.add_memlet(
            block3, dW_node3, "void", tasklet4, "_in1",
            {symbolic::add(symbolic::mul(symbolic::symbol("i"), symbolic::symbol("cols")),
                           symbolic::symbol("j"))});
        builder.add_memlet(
            block3, dW_node3, "void", tasklet4, "_in2",
            {symbolic::add(symbolic::mul(symbolic::symbol("i"), symbolic::symbol("cols")),
                           symbolic::symbol("j"))});
        builder.add_memlet(block3, G2_node3, "void", tasklet4, "_in3", {});
        builder.add_memlet(block3, tasklet4, "_out", G2_node3_out, "void", {});

        auto& block4 = builder.add_block(body_loop_j);
        auto& G2_node4 = builder.add_access(block4, "G2");
        auto& G2_node4_out = builder.add_access(block4, "G2");
        auto& dE_node4 = builder.add_access(block4, "dE");
        auto& tasklet5 = builder.add_tasklet(
            block4, data_flow::TaskletCode::fma, {"_out", element_desc},
            {{"_in1", element_desc}, {"_in2", element_desc}, {"_in3", element_desc}});
        builder.add_memlet(
            block4, dE_node4, "void", tasklet5, "_in1",
            {symbolic::add(symbolic::mul(symbolic::symbol("i"), symbolic::symbol("cols")),
                           symbolic::symbol("j"))});
        builder.add_memlet(
            block4, dE_node4, "void", tasklet5, "_in2",
            {symbolic::add(symbolic::mul(symbolic::symbol("i"), symbolic::symbol("cols")),
                           symbolic::symbol("j"))});
        builder.add_memlet(block4, G2_node4, "void", tasklet5, "_in3", {});
        builder.add_memlet(block4, tasklet5, "_out", G2_node4_out, "void", {});

        auto& block5 = builder.add_block(body_loop_j);
        auto& G2_node5 = builder.add_access(block5, "G2");
        auto& G2_node5_out = builder.add_access(block5, "G2");
        auto& Jc_node5 = builder.add_access(block5, "Jc");
        auto& tasklet6 =
            builder.add_tasklet(block5, data_flow::TaskletCode::div, {"_out", element_desc},
                                {{"_in1", element_desc}, {"_in2", element_desc}});
        builder.add_memlet(block5, G2_node5, "void", tasklet6, "_in1", {});
        builder.add_memlet(block5, Jc_node5, "void", tasklet6, "_in2", {});
        builder.add_memlet(block5, tasklet6, "_out", G2_node5_out, "void", {});
    }

    // L
    {
        auto& block1 = builder.add_block(body_loop_j);
        auto& L_node1 = builder.add_access(block1, "L");
        auto& dN_node1 = builder.add_access(block1, "dN");
        auto& dS_node1 = builder.add_access(block1, "dS");
        auto& tasklet1 =
            builder.add_tasklet(block1, data_flow::TaskletCode::add, {"_out", element_desc},
                                {{"_in1", element_desc}, {"_in2", element_desc}});
        builder.add_memlet(
            block1, dN_node1, "void", tasklet1, "_in1",
            {symbolic::add(symbolic::mul(symbolic::symbol("i"), symbolic::symbol("cols")),
                           symbolic::symbol("j"))});
        builder.add_memlet(
            block1, dS_node1, "void", tasklet1, "_in2",
            {symbolic::add(symbolic::mul(symbolic::symbol("i"), symbolic::symbol("cols")),
                           symbolic::symbol("j"))});
        builder.add_memlet(block1, tasklet1, "_out", L_node1, "void", {});

        auto& block2 = builder.add_block(body_loop_j);
        auto& L_node2 = builder.add_access(block2, "L");
        auto& L_node2_out = builder.add_access(block2, "L");
        auto& dW_node2 = builder.add_access(block2, "dW");
        auto& tasklet2 =
            builder.add_tasklet(block2, data_flow::TaskletCode::add, {"_out", element_desc},
                                {{"_in1", element_desc}, {"_in2", element_desc}});
        builder.add_memlet(block2, L_node2, "void", tasklet2, "_in1", {});
        builder.add_memlet(
            block2, dW_node2, "void", tasklet2, "_in2",
            {symbolic::add(symbolic::mul(symbolic::symbol("i"), symbolic::symbol("cols")),
                           symbolic::symbol("j"))});
        builder.add_memlet(block2, tasklet2, "_out", L_node2_out, "void", {});

        auto& block3 = builder.add_block(body_loop_j);
        auto& L_node3 = builder.add_access(block3, "L");
        auto& L_node3_out = builder.add_access(block3, "L");
        auto& dE_node3 = builder.add_access(block3, "dE");
        auto& tasklet3 =
            builder.add_tasklet(block3, data_flow::TaskletCode::add, {"_out", element_desc},
                                {{"_in1", element_desc}, {"_in2", element_desc}});
        builder.add_memlet(block3, L_node3, "void", tasklet3, "_in1", {});
        builder.add_memlet(
            block3, dE_node3, "void", tasklet3, "_in2",
            {symbolic::add(symbolic::mul(symbolic::symbol("i"), symbolic::symbol("cols")),
                           symbolic::symbol("j"))});
        builder.add_memlet(block3, tasklet3, "_out", L_node3_out, "void", {});

        auto& block4 = builder.add_block(body_loop_j);
        auto& L_node4 = builder.add_access(block4, "L");
        auto& L_node4_out = builder.add_access(block4, "L");
        auto& Jc_node4 = builder.add_access(block4, "Jc");
        auto& tasklet4 =
            builder.add_tasklet(block4, data_flow::TaskletCode::div, {"_out", element_desc},
                                {{"_in1", element_desc}, {"_in2", element_desc}});
        builder.add_memlet(block4, L_node4, "void", tasklet4, "_in1", {});
        builder.add_memlet(block4, Jc_node4, "void", tasklet4, "_in2", {});
        builder.add_memlet(block4, tasklet4, "_out", L_node4_out, "void", {});
    }

    // diffusion coefficent (equ 33)
    {
        auto& block1 = builder.add_block(body_loop_j);
        auto& c_node1 = builder.add_access(block1, "c");
        auto& L_node1 = builder.add_access(block1, "L");
        auto& q0sqr_node1 = builder.add_access(block1, "q0sqr");
        auto& tasklet1 =
            builder.add_tasklet(block1, data_flow::TaskletCode::sub, {"_out", element_desc},
                                {{"_in1", element_desc}, {"_in2", element_desc}});
        builder.add_memlet(block1, L_node1, "void", tasklet1, "_in1", {});
        builder.add_memlet(block1, q0sqr_node1, "void", tasklet1, "_in2", {});
        builder.add_memlet(
            block1, tasklet1, "_out", c_node1, "void",
            {symbolic::add(symbolic::mul(symbolic::symbol("i"), symbolic::symbol("cols")),
                           symbolic::symbol("j"))});
    }

    // saturate diffusion coefficent
    // if (c[i * cols + j] < 0) {c[i * cols + j] = 0;}
    // else if (c[i * cols + j] > 1) {c[i * cols + j] = 1;}
    {
        types::Scalar bool_desc(types::PrimitiveType::Bool);
        builder.add_container("tmp_0", bool_desc);
        builder.add_container("tmp_1", bool_desc);

        auto& block0 = builder.add_block(body_loop_j);
        auto& c_node0 = builder.add_access(block0, "c");
        auto& tmp_0_node = builder.add_access(block0, "tmp_0");
        auto& tasklet0 =
            builder.add_tasklet(block0, data_flow::TaskletCode::olt, {"_out", element_desc},
                                {{"_in", element_desc}, {"0", element_desc}});
        builder.add_memlet(
            block0, c_node0, "void", tasklet0, "_in",
            {symbolic::add(symbolic::mul(symbolic::symbol("i"), symbolic::symbol("cols")),
                           symbolic::symbol("j"))});
        builder.add_memlet(block0, tasklet0, "_out", tmp_0_node, "void", {});

        auto& block1 = builder.add_block(body_loop_j);
        auto& c_node1 = builder.add_access(block1, "c");
        auto& tmp_1_node = builder.add_access(block1, "tmp_1");
        auto& tasklet1 =
            builder.add_tasklet(block1, data_flow::TaskletCode::ogt, {"_out", element_desc},
                                {{"_in", element_desc}, {"1", element_desc}});
        builder.add_memlet(
            block1, c_node1, "void", tasklet1, "_in",
            {symbolic::add(symbolic::mul(symbolic::symbol("i"), symbolic::symbol("cols")),
                           symbolic::symbol("j"))});
        builder.add_memlet(block1, tasklet1, "_out", tmp_1_node, "void", {});

        auto& block2 = builder.add_block(body_loop_j);
        auto& c_node2 = builder.add_access(block2, "c");
        auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign,
                                             {"_out", element_desc}, {{"0", element_desc}});
        tasklet2.condition() = symbolic::Eq(symbolic::symbol("tmp_0"), symbolic::__true__());
        builder.add_memlet(
            block2, tasklet2, "_out", c_node2, "void",
            {symbolic::add(symbolic::mul(symbolic::symbol("i"), symbolic::symbol("cols")),
                           symbolic::symbol("j"))});

        auto& block3 = builder.add_block(body_loop_j);
        auto& c_node3 = builder.add_access(block3, "c");
        auto& tasklet3 = builder.add_tasklet(block3, data_flow::TaskletCode::assign,
                                             {"_out", element_desc}, {{"1", element_desc}});
        tasklet3.condition() = symbolic::Eq(symbolic::symbol("tmp_1"), symbolic::__true__());
        builder.add_memlet(
            block3, tasklet3, "_out", c_node3, "void",
            {symbolic::add(symbolic::mul(symbolic::symbol("i"), symbolic::symbol("cols")),
                           symbolic::symbol("j"))});
    }

    auto sdfg = builder.move();

    // Analysis
    analysis::AnalysisManager analysis_manager(*sdfg);
    auto& analysis = analysis_manager.get<analysis::DataParallelismAnalysis>();

    auto& dependencies1 = analysis.get(loop_i);
    EXPECT_EQ(dependencies1.size(), 22);
    EXPECT_EQ(dependencies1.at("cols"), analysis::Parallelism::READONLY);
    EXPECT_EQ(dependencies1.at("q0sqr"), analysis::Parallelism::READONLY);
    EXPECT_EQ(dependencies1.at("J"), analysis::Parallelism::READONLY);
    EXPECT_EQ(dependencies1.at("iN"), analysis::Parallelism::READONLY);
    EXPECT_EQ(dependencies1.at("iS"), analysis::Parallelism::READONLY);
    EXPECT_EQ(dependencies1.at("jW"), analysis::Parallelism::READONLY);
    EXPECT_EQ(dependencies1.at("jE"), analysis::Parallelism::READONLY);

    EXPECT_EQ(dependencies1.at("iN_i"), analysis::Parallelism::PRIVATE);
    EXPECT_EQ(dependencies1.at("iS_i"), analysis::Parallelism::PRIVATE);
    EXPECT_EQ(dependencies1.at("jW_j"), analysis::Parallelism::PRIVATE);
    EXPECT_EQ(dependencies1.at("jE_j"), analysis::Parallelism::PRIVATE);

    EXPECT_EQ(dependencies1.at("j"), analysis::Parallelism::PRIVATE);
    EXPECT_EQ(dependencies1.at("Jc"), analysis::Parallelism::PRIVATE);
    EXPECT_EQ(dependencies1.at("G2"), analysis::Parallelism::PRIVATE);
    EXPECT_EQ(dependencies1.at("L"), analysis::Parallelism::PRIVATE);

    EXPECT_EQ(dependencies1.at("tmp_0"), analysis::Parallelism::PRIVATE);
    EXPECT_EQ(dependencies1.at("tmp_1"), analysis::Parallelism::PRIVATE);

    EXPECT_EQ(dependencies1.at("dN"), analysis::Parallelism::PARALLEL);
    EXPECT_EQ(dependencies1.at("dS"), analysis::Parallelism::PARALLEL);
    EXPECT_EQ(dependencies1.at("dW"), analysis::Parallelism::PARALLEL);
    EXPECT_EQ(dependencies1.at("dE"), analysis::Parallelism::PARALLEL);
    EXPECT_EQ(dependencies1.at("c"), analysis::Parallelism::PARALLEL);

    auto& dependencies2 = analysis.get(loop_j);
    EXPECT_EQ(dependencies2.size(), 22);
    EXPECT_EQ(dependencies2.at("cols"), analysis::Parallelism::READONLY);
    EXPECT_EQ(dependencies2.at("q0sqr"), analysis::Parallelism::READONLY);
    EXPECT_EQ(dependencies2.at("J"), analysis::Parallelism::READONLY);
    EXPECT_EQ(dependencies2.at("iN"), analysis::Parallelism::READONLY);
    EXPECT_EQ(dependencies2.at("iS"), analysis::Parallelism::READONLY);
    EXPECT_EQ(dependencies2.at("jW"), analysis::Parallelism::READONLY);
    EXPECT_EQ(dependencies2.at("jE"), analysis::Parallelism::READONLY);
    EXPECT_EQ(dependencies2.at("i"), analysis::Parallelism::READONLY);

    EXPECT_EQ(dependencies2.at("iN_i"), analysis::Parallelism::PRIVATE);
    EXPECT_EQ(dependencies2.at("iS_i"), analysis::Parallelism::PRIVATE);
    EXPECT_EQ(dependencies2.at("jW_j"), analysis::Parallelism::PRIVATE);
    EXPECT_EQ(dependencies2.at("jE_j"), analysis::Parallelism::PRIVATE);

    EXPECT_EQ(dependencies2.at("Jc"), analysis::Parallelism::PRIVATE);
    EXPECT_EQ(dependencies2.at("G2"), analysis::Parallelism::PRIVATE);
    EXPECT_EQ(dependencies2.at("L"), analysis::Parallelism::PRIVATE);

    EXPECT_EQ(dependencies2.at("tmp_0"), analysis::Parallelism::PRIVATE);
    EXPECT_EQ(dependencies2.at("tmp_1"), analysis::Parallelism::PRIVATE);

    EXPECT_EQ(dependencies2.at("dN"), analysis::Parallelism::PARALLEL);
    EXPECT_EQ(dependencies2.at("dS"), analysis::Parallelism::PARALLEL);
    EXPECT_EQ(dependencies2.at("dW"), analysis::Parallelism::PARALLEL);
    EXPECT_EQ(dependencies2.at("dE"), analysis::Parallelism::PARALLEL);
    EXPECT_EQ(dependencies2.at("c"), analysis::Parallelism::PARALLEL);
}
