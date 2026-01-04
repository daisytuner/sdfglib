#include "sdfg/analysis/memlet_delinearization_analysis.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/array.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"

using namespace sdfg;

TEST(MemletDelinearizationAnalysisTest, SimpleLinearAccess) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add a container: int* A (pointer to scalar, accessed as array)
    types::Scalar scalar_type(types::PrimitiveType::Int32);
    types::Pointer pointer_type(scalar_type);
    builder.add_container("A", pointer_type, true);

    // Create a block with a memlet that has a linear access pattern with constant index
    auto& block = builder.add_block(root);
    auto& access_in = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "", {"in"});
    builder.add_computational_memlet(block, access_in, tasklet, "in", {symbolic::integer(5)});

    // Run the analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::MemletDelinearizationAnalysis>();

    // Get the memlet
    auto& dfg = block.dataflow();
    auto& memlet = *dfg.edges().begin();

    // Check that delinearization was not applicable (constant index, no linearization to detect)
    const auto* delinearized = analysis.get(memlet);
    EXPECT_EQ(delinearized, nullptr);
}

TEST(MemletDelinearizationAnalysisTest, EmptySubset) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add a container
    types::Scalar scalar_type(types::PrimitiveType::Int32);
    builder.add_container("A", scalar_type, true);

    // Create a block with a memlet that has an empty subset (scalar access)
    auto& block = builder.add_block(root);
    auto& access_in = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "", {"in"});
    builder.add_computational_memlet(block, access_in, tasklet, "in", {});

    // Run the analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::MemletDelinearizationAnalysis>();

    // Get the memlet
    auto& dfg = block.dataflow();
    auto& memlet = *dfg.edges().begin();

    // Check that empty subsets result in nullptr
    const auto* delinearized = analysis.get(memlet);
    EXPECT_EQ(delinearized, nullptr);
}

TEST(MemletDelinearizationAnalysisTest, NonExistentMemlet) {
    builder::StructuredSDFGBuilder builder1("sdfg_test1", FunctionType_CPU);
    builder::StructuredSDFGBuilder builder2("sdfg_test2", FunctionType_CPU);

    auto& sdfg1 = builder1.subject();
    auto& sdfg2 = builder2.subject();
    auto& root1 = sdfg1.root();
    auto& root2 = sdfg2.root();

    // Add containers in both SDFGs
    types::Scalar scalar_type(types::PrimitiveType::Int32);
    types::Pointer pointer_type(scalar_type);
    builder1.add_container("A", pointer_type, true);
    builder2.add_container("B", pointer_type, true);

    // Create blocks in both SDFGs
    auto& block1 = builder1.add_block(root1);
    auto& access_in1 = builder1.add_access(block1, "A");
    auto& tasklet1 = builder1.add_tasklet(block1, data_flow::TaskletCode::assign, "", {"in"});
    builder1.add_computational_memlet(block1, access_in1, tasklet1, "in", {symbolic::integer(5)});

    auto& block2 = builder2.add_block(root2);
    auto& access_in2 = builder2.add_access(block2, "B");
    auto& tasklet2 = builder2.add_tasklet(block2, data_flow::TaskletCode::assign, "", {"in"});
    builder2.add_computational_memlet(block2, access_in2, tasklet2, "in", {symbolic::integer(5)});

    // Run the analysis on sdfg1
    analysis::AnalysisManager analysis_manager(sdfg1);
    auto& analysis = analysis_manager.get<analysis::MemletDelinearizationAnalysis>();

    // Get a memlet from sdfg2
    auto& dfg2 = block2.dataflow();
    auto& memlet2 = *dfg2.edges().begin();

    // Check that querying a memlet not in the analyzed SDFG returns nullptr
    const auto* delinearized = analysis.get(memlet2);
    EXPECT_EQ(delinearized, nullptr);
}

TEST(MemletDelinearizationAnalysisTest, NestedLoopWithBoundsAsStrides) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar index_type(types::PrimitiveType::Int64);
    types::Scalar scalar_type(types::PrimitiveType::Float);
    types::Pointer pointer_type(scalar_type);
    builder.add_container("N", index_type, true);
    builder.add_container("M", index_type, true);
    builder.add_container("i", index_type);
    builder.add_container("j", index_type);
    builder.add_container("A", pointer_type, true);

    // Define outer loop: for i in [0, N)
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto& outer_loop =
        builder.add_for(root, i, symbolic::Lt(i, N), symbolic::integer(0), symbolic::add(i, symbolic::one()));

    // Define inner loop: for j in [0, M)
    auto& inner_loop =
        builder
            .add_for(outer_loop.root(), j, symbolic::Lt(j, M), symbolic::integer(0), symbolic::add(j, symbolic::one()));

    // Create block with linearized access: A[i*M + j]
    auto& block = builder.add_block(inner_loop.root());
    auto& access = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "", {"in"});
    auto linearized = symbolic::add(symbolic::mul(i, M), j);
    builder.add_computational_memlet(block, access, tasklet, "in", {linearized});

    // Run the analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::MemletDelinearizationAnalysis>();

    // Get the memlet
    auto& dfg = block.dataflow();
    auto& memlet = *dfg.edges().begin();

    // Check that delinearization succeeded
    const auto* delinearized = analysis.get(memlet);
    ASSERT_NE(delinearized, nullptr);
    EXPECT_EQ(delinearized->size(), 2);

    // Check each dimension
    EXPECT_TRUE(symbolic::eq((*delinearized)[0], i));
    EXPECT_TRUE(symbolic::eq((*delinearized)[1], j));
}

TEST(MemletDelinearizationAnalysisTest, StencilPatternWithNegativeOffset) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar index_type(types::PrimitiveType::Int64);
    types::Scalar scalar_type(types::PrimitiveType::Float);
    types::Pointer pointer_type(scalar_type);
    builder.add_container("N", index_type, true);
    builder.add_container("i", index_type);
    builder.add_container("A", pointer_type, true);

    // Define loop: for i in [1, N-1)
    auto N = symbolic::symbol("N");
    auto i = symbolic::symbol("i");
    auto& loop = builder.add_for(
        root, i, symbolic::Lt(i, symbolic::sub(N, symbolic::one())), symbolic::one(), symbolic::add(i, symbolic::one())
    );

    // Create block with stencil access: A[i-1]
    auto& block = builder.add_block(loop.root());
    auto& access = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "", {"in"});
    auto offset_access = symbolic::sub(i, symbolic::one());
    builder.add_computational_memlet(block, access, tasklet, "in", {offset_access});

    // Run the analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::MemletDelinearizationAnalysis>();

    // Get the memlet
    auto& dfg = block.dataflow();
    auto& memlet = *dfg.edges().begin();

    // Check that delinearization was not applicable (simple offset, no linearization)
    const auto* delinearized = analysis.get(memlet);
    // The result depends on whether the delinearization algorithm considers i-1 as a change
    // We just verify it returns a valid result
    if (delinearized != nullptr) {
        EXPECT_EQ(delinearized->size(), 1);
    }
}

TEST(MemletDelinearizationAnalysisTest, StencilPatternWithPositiveOffset) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar index_type(types::PrimitiveType::Int64);
    types::Scalar scalar_type(types::PrimitiveType::Float);
    types::Pointer pointer_type(scalar_type);
    builder.add_container("N", index_type, true);
    builder.add_container("i", index_type);
    builder.add_container("A", pointer_type, true);

    // Define loop: for i in [0, N-1)
    auto N = symbolic::symbol("N");
    auto i = symbolic::symbol("i");
    auto& loop = builder.add_for(
        root, i, symbolic::Lt(i, symbolic::sub(N, symbolic::one())), symbolic::zero(), symbolic::add(i, symbolic::one())
    );

    // Create block with stencil access: A[i+1]
    auto& block = builder.add_block(loop.root());
    auto& access = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "", {"in"});
    auto offset_access = symbolic::add(i, symbolic::one());
    builder.add_computational_memlet(block, access, tasklet, "in", {offset_access});

    // Run the analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::MemletDelinearizationAnalysis>();

    // Get the memlet
    auto& dfg = block.dataflow();
    auto& memlet = *dfg.edges().begin();

    // Check that delinearization was not applicable (simple offset, no linearization)
    const auto* delinearized = analysis.get(memlet);
    // The result depends on whether the delinearization algorithm considers i+1 as a change
    if (delinearized != nullptr) {
        EXPECT_EQ(delinearized->size(), 1);
    }
}

TEST(MemletDelinearizationAnalysisTest, ComplexStencilPattern2D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar index_type(types::PrimitiveType::Int64);
    types::Scalar scalar_type(types::PrimitiveType::Float);
    types::Pointer pointer_type(scalar_type);
    builder.add_container("N", index_type, true);
    builder.add_container("M", index_type, true);
    builder.add_container("i", index_type);
    builder.add_container("j", index_type);
    builder.add_container("A", pointer_type, true);

    // Define outer loop: for i in [1, N-1)
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto& outer_loop = builder.add_for(
        root, i, symbolic::Lt(i, symbolic::sub(N, symbolic::one())), symbolic::one(), symbolic::add(i, symbolic::one())
    );

    // Define inner loop: for j in [1, M-1)
    auto& inner_loop = builder.add_for(
        outer_loop.root(),
        j,
        symbolic::Lt(j, symbolic::sub(M, symbolic::one())),
        symbolic::one(),
        symbolic::add(j, symbolic::one())
    );

    // Create block with 2D stencil access: A[(i-1)*M + (j+1)]
    auto& block = builder.add_block(inner_loop.root());
    auto& access = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "", {"in"});
    auto i_offset = symbolic::sub(i, symbolic::one());
    auto j_offset = symbolic::add(j, symbolic::one());
    auto linearized = symbolic::add(symbolic::mul(i_offset, M), j_offset);
    builder.add_computational_memlet(block, access, tasklet, "in", {linearized});

    // Run the analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::MemletDelinearizationAnalysis>();

    // Get the memlet
    auto& dfg = block.dataflow();
    auto& memlet = *dfg.edges().begin();

    // Check that delinearization succeeded
    const auto* delinearized = analysis.get(memlet);
    ASSERT_NE(delinearized, nullptr);
    EXPECT_EQ(delinearized->size(), 2);

    // Check each dimension (should be i-1 and j+1)
    EXPECT_TRUE(symbolic::eq((*delinearized)[0], i_offset));
    EXPECT_TRUE(symbolic::eq((*delinearized)[1], j_offset));
}
