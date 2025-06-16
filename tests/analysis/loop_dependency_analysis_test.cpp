#include "sdfg/analysis/loop_dependency_analysis.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"

using namespace sdfg;

TEST(LoopDependencyAnalysisTest, Last_1D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("B", base_desc, true);

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
    auto& b_out = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", base_desc},
                                        {{"_in", base_desc}});
    builder.add_memlet(block, a1, "void", tasklet, "_in", {indvar});
    builder.add_memlet(block, tasklet, "_out", b_out, "void", {});

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::LoopDependencyAnalysis>();
    auto dependencies = analysis.get(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 1);
    EXPECT_EQ(dependencies.at("B"), analysis::LoopCarriedDependency::WAW);
}

TEST(LoopDependencyAnalysisTest, Sum_1D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("B", base_desc, true);

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
    auto& b_in = builder.add_access(block, "B");
    auto& b_out = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::add, {"_out", base_desc},
                                        {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block, a1, "void", tasklet, "_in1", {indvar});
    builder.add_memlet(block, b_in, "void", tasklet, "_in2", {});
    builder.add_memlet(block, tasklet, "_out", b_out, "void", {});

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::LoopDependencyAnalysis>();
    auto dependencies = analysis.get(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 1);
    EXPECT_EQ(dependencies.at("B"), analysis::LoopCarriedDependency::RAW);
}

TEST(LoopDependencyAnalysisTest, Shift_1D) {
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
    auto& a1 = builder.add_access(block, "A");
    auto& a2 = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", base_desc},
                                        {{"_in", base_desc}});
    builder.add_memlet(block, a1, "void", tasklet, "_in",
                       {symbolic::sub(indvar, symbolic::integer(1))});
    builder.add_memlet(block, tasklet, "_out", a2, "void", {indvar});

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::LoopDependencyAnalysis>();
    auto dependencies = analysis.get(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 1);
    EXPECT_EQ(dependencies.at("A"), analysis::LoopCarriedDependency::RAW);
}

TEST(LoopDependencyAnalysisTest, PartialSum_1D) {
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

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::LoopDependencyAnalysis>();
    auto dependencies = analysis.get(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 1);
    EXPECT_EQ(dependencies.at("A"), analysis::LoopCarriedDependency::RAW);
}

TEST(LoopDependencyAnalysisTest, LoopLocal_1D) {
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
    auto& tasklet_1 = builder.add_tasklet(block_1, data_flow::TaskletCode::assign,
                                          {"_out", sym_desc}, {{"_in", sym_desc}});
    builder.add_memlet(block_1, i_in, "void", tasklet_1, "_in", {});
    builder.add_memlet(block_1, tasklet_1, "_out", tmp_out, "void", {});

    auto& block_2 = builder.add_block(body);
    auto& tmp_in = builder.add_access(block_2, "tmp");
    auto& a_out = builder.add_access(block_2, "A");
    auto& tasklet = builder.add_tasklet(block_2, data_flow::TaskletCode::assign,
                                        {"_out", base_desc}, {{"_in", base_desc}});
    builder.add_memlet(block_2, tmp_in, "void", tasklet, "_in", {});
    builder.add_memlet(block_2, tasklet, "_out", a_out, "void", {indvar});

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::LoopDependencyAnalysis>();
    auto dependencies = analysis.get(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 1);
    EXPECT_EQ(dependencies.at("tmp"), analysis::LoopCarriedDependency::WAW);
}

TEST(LoopDependencyAnalysisTest, Store_1D) {
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
    auto& a = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", base_desc},
                                        {{"0", base_desc}});
    builder.add_memlet(block, tasklet, "_out", a, "void", {indvar});

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::LoopDependencyAnalysis>();
    auto dependencies = analysis.get(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 0);
}

TEST(LoopDependencyAnalysisTest, Copy_1D) {
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
    auto& a = builder.add_access(block, "A");
    auto& b = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", base_desc},
                                        {{"_in", base_desc}});
    builder.add_memlet(block, a, "void", tasklet, "_in", {indvar});
    builder.add_memlet(block, tasklet, "_out", b, "void", {indvar});

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::LoopDependencyAnalysis>();
    auto dependencies = analysis.get(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 0);
}

TEST(LoopDependencyAnalysisTest, Map_1D) {
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
    auto& a_in = builder.add_access(block, "A");
    auto& i = builder.add_access(block, "i");
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::add, {"_out", base_desc},
                                        {{"_in1", base_desc}, {"_in2", sym_desc}});
    builder.add_memlet(block, a_in, "void", tasklet, "_in1", {indvar});
    builder.add_memlet(block, i, "void", tasklet, "_in2", {});
    builder.add_memlet(block, tasklet, "_out", a_out, "void", {indvar});

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::LoopDependencyAnalysis>();
    auto dependencies = analysis.get(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 0);
}

TEST(LoopDependencyAnalysisTest, MapParameterized_1D) {
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
    builder.add_container("m", sym_desc);
    builder.add_container("b", sym_desc);

    auto& assums_m = sdfg.assumption(symbolic::symbol("m"));
    assums_m.lower_bound(symbolic::integer(1));
    auto& assums_b = sdfg.assumption(symbolic::symbol("b"));
    assums_b.lower_bound(symbolic::integer(1));

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
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", base_desc},
                                        {{"_in1", base_desc}});
    builder.add_memlet(
        block, A_in, "void", tasklet, "_in1",
        {symbolic::add(symbolic::mul(symbolic::symbol("m"), indvar), symbolic::symbol("b"))});
    builder.add_memlet(
        block, tasklet, "_out", A_out, "void",
        {symbolic::add(symbolic::mul(symbolic::symbol("m"), indvar), symbolic::symbol("b"))});

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::LoopDependencyAnalysis>();
    auto dependencies = analysis.get(loop);

    // Check
    std::cout << dependencies.size() << std::endl;
    for (auto& dependency : dependencies) {
        std::cout << dependency.first << " " << dependency.second << std::endl;
    }
    EXPECT_EQ(dependencies.size(), 0);
}

TEST(LoopDependencyAnalysisTest, Stencil_1D) {
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

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::LoopDependencyAnalysis>();
    auto dependencies = analysis.get(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 0);
}

TEST(LoopDependencyAnalysisTest, Gather_1D) {
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
    auto& block_1 = builder.add_block(body);

    // Define indirection
    auto& A = builder.add_access(block_1, "A");
    auto& b = builder.add_access(block_1, "b");
    auto& tasklet1 = builder.add_tasklet(block_1, data_flow::TaskletCode::assign,
                                         {"_out", base_desc}, {{"_in", base_desc}});
    builder.add_memlet(block_1, A, "void", tasklet1, "_in", {indvar});
    builder.add_memlet(block_1, tasklet1, "_out", b, "void", {});

    auto& block_2 = builder.add_block(body);
    auto& B = builder.add_access(block_2, "B");
    auto& C = builder.add_access(block_2, "C");
    auto& tasklet = builder.add_tasklet(block_2, data_flow::TaskletCode::assign,
                                        {"_out", base_desc}, {{"_in", base_desc}});
    builder.add_memlet(block_2, B, "void", tasklet, "_in", {symbolic::symbol("b")});
    builder.add_memlet(block_2, tasklet, "_out", C, "void", {indvar});

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::LoopDependencyAnalysis>();
    auto dependencies = analysis.get(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 1);
    EXPECT_EQ(dependencies.at("b"), analysis::LoopCarriedDependency::WAW);
}

TEST(LoopDependencyAnalysisTest, Scatter_1D) {
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

    // Define indirection
    auto& block_1 = builder.add_block(body);
    auto& A = builder.add_access(block_1, "A");
    auto& b = builder.add_access(block_1, "b");
    auto& tasklet1 = builder.add_tasklet(block_1, data_flow::TaskletCode::assign,
                                         {"_out", base_desc}, {{"_in", base_desc}});
    builder.add_memlet(block_1, A, "void", tasklet1, "_in", {indvar});
    builder.add_memlet(block_1, tasklet1, "_out", b, "void", {});

    auto& block_2 = builder.add_block(body);
    auto& B = builder.add_access(block_2, "B");
    auto& C = builder.add_access(block_2, "C");
    auto& tasklet = builder.add_tasklet(block_2, data_flow::TaskletCode::assign,
                                        {"_out", base_desc}, {{"_in", base_desc}});
    builder.add_memlet(block_2, B, "void", tasklet, "_in", {indvar});
    builder.add_memlet(block_2, tasklet, "_out", C, "void", {symbolic::symbol("b")});

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::LoopDependencyAnalysis>();
    auto dependencies = analysis.get(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 2);
    EXPECT_EQ(dependencies.at("b"), analysis::LoopCarriedDependency::WAW);
    EXPECT_EQ(dependencies.at("C"), analysis::LoopCarriedDependency::WAW);
}

TEST(LoopDependencyAnalysisTest, MapDeg2_1D) {
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
    builder.add_memlet(block, tasklet, "_out", A, "void", {symbolic::mul(indvar, indvar)});

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::LoopDependencyAnalysis>();
    auto dependencies = analysis.get(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 0);
}

TEST(LoopDependencyAnalysisTest, Map_2D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Pointer desc_2(static_cast<const types::IType&>(desc));
    builder.add_container("A", desc_2, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
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
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::add, {"_out", base_desc},
                                        {{"_in1", base_desc}, {"_in2", sym_desc}});
    builder.add_memlet(block, a_in, "void", tasklet, "_in1", {indvar, indvar_2});
    builder.add_memlet(block, i, "void", tasklet, "_in2", {});
    builder.add_memlet(block, tasklet, "_out", a_out, "void", {indvar, indvar_2});

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::LoopDependencyAnalysis>();

    // Check
    auto dependencies = analysis.get(loop);
    EXPECT_EQ(dependencies.size(), 1);
    EXPECT_EQ(dependencies.at("j"), analysis::LoopCarriedDependency::WAW);

    // Check loop 2
    dependencies = analysis.get(loop_2);
    EXPECT_EQ(dependencies.size(), 0);
}

TEST(LoopDependencyAnalysisTest, PartialSumInner_2D) {
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

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::LoopDependencyAnalysis>();
    auto dependencies1 = analysis.get(loop1);
    auto dependencies2 = analysis.get(loop2);

    // Check
    EXPECT_EQ(dependencies1.size(), 1);
    EXPECT_EQ(dependencies1.at("j"), analysis::LoopCarriedDependency::WAW);

    EXPECT_EQ(dependencies2.size(), 1);
    EXPECT_EQ(dependencies2.at("B"), analysis::LoopCarriedDependency::RAW);
}

TEST(LoopDependencyAnalysisTest, PartialSumOuter_2D) {
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

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::LoopDependencyAnalysis>();
    auto dependencies1 = analysis.get(loop1);
    auto dependencies2 = analysis.get(loop2);

    // Check
    EXPECT_EQ(dependencies1.size(), 2);
    EXPECT_EQ(dependencies1.at("j"), analysis::LoopCarriedDependency::WAW);
    EXPECT_EQ(dependencies1.at("B"), analysis::LoopCarriedDependency::RAW);

    EXPECT_EQ(dependencies2.size(), 0);
}

TEST(LoopDependencyAnalysisTest, Transpose_2D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Pointer desc2(static_cast<const types::IType&>(desc));
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
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", base_desc},
                                        {{"_in", base_desc}});
    builder.add_memlet(block, A, "void", tasklet, "_in", {indvar1, indvar2});
    builder.add_memlet(block, tasklet, "_out", B, "void", {indvar2, indvar1});

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::LoopDependencyAnalysis>();
    auto dependencies1 = analysis.get(loop1);
    auto dependencies2 = analysis.get(loop2);

    // Check
    EXPECT_EQ(dependencies2.size(), 0);
    EXPECT_EQ(dependencies1.size(), 1);
    EXPECT_EQ(dependencies1.at("j"), analysis::LoopCarriedDependency::WAW);
}

TEST(LoopDependencyAnalysisTest, TransposeTriangle_2D) {
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
    types::Pointer desc2(static_cast<const types::IType&>(desc));
    builder.add_container("A", desc2, true);

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
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", base_desc},
                                        {{"_in", base_desc}});
    builder.add_memlet(block, A_in, "void", tasklet, "_in", {indvar1, indvar2});
    builder.add_memlet(block, tasklet, "_out", A_out, "void", {indvar2, indvar1});

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::LoopDependencyAnalysis>();
    auto dependencies1 = analysis.get(loop1);
    auto dependencies2 = analysis.get(loop2);

    // Check
    EXPECT_EQ(dependencies1.size(), 1);
    EXPECT_EQ(dependencies1.at("j"), analysis::LoopCarriedDependency::WAW);
    EXPECT_EQ(dependencies2.size(), 0);
}

TEST(LoopDependencyAnalysisTest, TransposeTriangleWithDiagonal_2D) {
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
    types::Pointer desc2(static_cast<const types::IType&>(desc));
    builder.add_container("A", desc2, true);

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
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", base_desc},
                                        {{"_in", base_desc}});
    builder.add_memlet(block, A_in, "void", tasklet, "_in", {indvar1, indvar2});
    builder.add_memlet(block, tasklet, "_out", A_out, "void", {indvar2, indvar1});

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::LoopDependencyAnalysis>();
    auto dependencies1 = analysis.get(loop1);
    auto dependencies2 = analysis.get(loop2);

    // Check
    EXPECT_EQ(dependencies1.size(), 1);
    EXPECT_EQ(dependencies1.at("j"), analysis::LoopCarriedDependency::WAW);
    EXPECT_EQ(dependencies2.size(), 0);
}

TEST(LoopDependencyAnalysisTest, TransposeSquare_2D) {
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
    types::Pointer desc2(static_cast<const types::IType&>(desc));
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
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", base_desc},
                                        {{"_in", base_desc}});
    builder.add_memlet(block, A_in, "void", tasklet, "_in", {indvar1, indvar2});
    builder.add_memlet(block, tasklet, "_out", A_out, "void", {indvar2, indvar1});

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::LoopDependencyAnalysis>();
    auto dependencies1 = analysis.get(loop1);
    auto dependencies2 = analysis.get(loop2);

    // Check
    EXPECT_EQ(dependencies2.size(), 0);
    EXPECT_EQ(dependencies1.size(), 2);
    EXPECT_EQ(dependencies1.at("A"), analysis::LoopCarriedDependency::RAW);
    EXPECT_EQ(dependencies1.at("j"), analysis::LoopCarriedDependency::WAW);
}
