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
    auto& root = builder.root();

    // Add a container: int A[10]
    types::Scalar scalar_type(types::PrimitiveType::Int32);
    types::Array array_type(scalar_type, symbolic::integer(10));
    types::Pointer pointer_type(
        types::StorageType::Default, types::AlignmentType::None, types::InitializerType::None, array_type
    );
    builder.add_container("A", pointer_type, true);

    // Create a block with a memlet that has a linear access pattern
    auto& block = builder.add_block(root);
    auto& access_in = builder.add_access_node(block, "A");
    auto& tasklet = builder.add_tasklet(block, {"in"}, {});
    auto i = symbolic::symbol("i");
    builder.add_computational_memlet(block, access_in, "out", tasklet, "in", {i}, scalar_type);

    // Run the analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::MemletDelinearizationAnalysis>();

    // Get the memlet
    auto& dfg = block.dataflow();
    auto& memlet = *dfg.edges().begin();

    // Check that delinearization was not applicable (already linear)
    const auto* delinearized = analysis.get(memlet);
    EXPECT_EQ(delinearized, nullptr);
}

TEST(MemletDelinearizationAnalysisTest, LinearizedAccess) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = builder.root();

    // Add a container: int A[10][20]
    types::Scalar scalar_type(types::PrimitiveType::Int32);
    types::Array inner_array(scalar_type, symbolic::integer(20));
    types::Array outer_array(inner_array, symbolic::integer(10));
    types::Pointer pointer_type(
        types::StorageType::Default, types::AlignmentType::None, types::InitializerType::None, outer_array
    );
    builder.add_container("A", pointer_type, true);

    // Create a block with a memlet that has a linearized access: i*20 + j
    auto& block = builder.add_block(root);
    auto& access_in = builder.add_access_node(block, "A");
    auto& tasklet = builder.add_tasklet(block, {"in"}, {});
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto linearized = symbolic::add(symbolic::mul(i, symbolic::integer(20)), j);
    builder.add_computational_memlet(block, access_in, "out", tasklet, "in", {linearized}, scalar_type);

    // Run the analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::MemletDelinearizationAnalysis>();

    // Get the memlet
    auto& dfg = block.dataflow();
    auto& memlet = *dfg.edges().begin();

    // Check that delinearization was successful
    const auto* delinearized = analysis.get(memlet);
    ASSERT_NE(delinearized, nullptr);
    EXPECT_EQ(delinearized->size(), 2);
}

TEST(MemletDelinearizationAnalysisTest, EmptySubset) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = builder.root();

    // Add a container
    types::Scalar scalar_type(types::PrimitiveType::Int32);
    builder.add_container("A", scalar_type, true);

    // Create a block with a memlet that has an empty subset (scalar access)
    auto& block = builder.add_block(root);
    auto& access_in = builder.add_access_node(block, "A");
    auto& tasklet = builder.add_tasklet(block, {"in"}, {});
    builder.add_computational_memlet(block, access_in, "out", tasklet, "in", {}, scalar_type);

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
    auto& root1 = builder1.root();
    auto& root2 = builder2.root();

    // Add containers in both SDFGs
    types::Scalar scalar_type(types::PrimitiveType::Int32);
    types::Array array_type(scalar_type, symbolic::integer(10));
    types::Pointer pointer_type(
        types::StorageType::Default, types::AlignmentType::None, types::InitializerType::None, array_type
    );
    builder1.add_container("A", pointer_type, true);
    builder2.add_container("B", pointer_type, true);

    // Create blocks in both SDFGs
    auto& block1 = builder1.add_block(root1);
    auto& access_in1 = builder1.add_access_node(block1, "A");
    auto& tasklet1 = builder1.add_tasklet(block1, {"in"}, {});
    auto i = symbolic::symbol("i");
    builder1.add_computational_memlet(block1, access_in1, "out", tasklet1, "in", {i}, scalar_type);

    auto& block2 = builder2.add_block(root2);
    auto& access_in2 = builder2.add_access_node(block2, "B");
    auto& tasklet2 = builder2.add_tasklet(block2, {"in"}, {});
    builder2.add_computational_memlet(block2, access_in2, "out", tasklet2, "in", {i}, scalar_type);

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
