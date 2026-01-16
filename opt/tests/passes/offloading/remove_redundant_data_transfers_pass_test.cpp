#include <gtest/gtest.h>
#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/cuda/cuda_offloading_node.h"
#include "sdfg/element.h"
#include "sdfg/function.h"
#include "sdfg/memory/offloading_node.h"
#include "sdfg/passes/remove_redundant_transfers_pass.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/symbolic/symbolic.h"

using namespace sdfg;

TEST(RemoveRedundantTransfersPassTest, SingleTransferTest) {
    sdfg::builder::StructuredSDFGBuilder builder("dot", sdfg::FunctionType_CPU);
    sdfg::analysis::AnalysisManager analysis_manager(builder.subject());

    auto& root = builder.subject().root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("__daisy_offload_A", desc);

    auto& block = builder.add_block(root);
    auto& access_node_in = builder.add_access(block, "__daisy_offload_A");
    auto& access_node_out = builder.add_access(block, "A");

    auto& memcpy_node = builder.add_library_node<cuda::CUDAOffloadingNode>(
        block,
        DebugInfo(),
        symbolic::integer(400),
        symbolic::integer(0),
        memory::DataTransferDirection::D2H,
        memory::BufferLifecycle::NO_CHANGE
    );

    auto& in_type = builder.subject().type("A");
    builder.add_computational_memlet(block, access_node_in, memcpy_node, "_src", {}, in_type);

    auto& out_type = builder.subject().type("A");
    builder.add_computational_memlet(block, memcpy_node, "_dst", access_node_out, {}, out_type);

    passes::RemoveRedundantTransfersPass pass;
    EXPECT_FALSE(pass.run_pass(builder, analysis_manager));
};

TEST(RemoveRedundantTransfersPassTest, MultiMapTest) {
    sdfg::builder::StructuredSDFGBuilder builder("dot", sdfg::FunctionType_CPU);
    sdfg::analysis::AnalysisManager analysis_manager(builder.subject());

    auto& root = builder.subject().root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("__daisy_offload_A", desc);

    auto& block = builder.add_block(root);
    auto& access_node_in = builder.add_access(block, "_daisy_offload_A");
    auto& access_node_out = builder.add_access(block, "A");

    auto& memcpy_node = builder.add_library_node<cuda::CUDAOffloadingNode>(
        block,
        DebugInfo(),
        symbolic::integer(400),
        symbolic::integer(0),
        memory::DataTransferDirection::D2H,
        memory::BufferLifecycle::NO_CHANGE
    );

    auto& in_type = builder.subject().type("A");
    builder.add_computational_memlet(block, access_node_in, memcpy_node, "_src", {}, in_type);

    auto& out_type = builder.subject().type("A");
    builder.add_computational_memlet(block, memcpy_node, "_dst", access_node_out, {}, out_type);

    auto& block2 = builder.add_block(root);

    auto& access_node_in2 = builder.add_access(block2, "__daisy_offload_A");
    auto& access_node_out2 = builder.add_access(block2, "A");

    auto& memcpy_node2 = builder.add_library_node<cuda::CUDAOffloadingNode>(
        block2,
        DebugInfo(),
        symbolic::integer(400),
        symbolic::integer(0),
        memory::DataTransferDirection::D2H,
        memory::BufferLifecycle::NO_CHANGE
    );

    auto& in_type2 = builder.subject().type("A");
    builder.add_computational_memlet(block2, access_node_in2, memcpy_node2, "_src", {}, in_type2);

    auto& out_type2 = builder.subject().type("A");
    builder.add_computational_memlet(block2, memcpy_node2, "_dst", access_node_out2, {}, out_type2);

    passes::RemoveRedundantTransfersPass pass;
    EXPECT_TRUE(pass.run_pass(builder, analysis_manager));

    int d2h_count = 0;
    for (int i = 0; i < root.size(); i++) {
        auto& cf_node = root.at(i).first;
        if (auto* block = dynamic_cast<structured_control_flow::Block*>(&cf_node)) {
            for (auto& node : block->dataflow().nodes()) {
                if (auto* data_transfer = dynamic_cast<memory::OffloadingNode*>(&node)) {
                    if (data_transfer->is_d2h()) {
                        d2h_count++;
                    }
                }
            }
        }
    }
    EXPECT_EQ(d2h_count, 1);
};

TEST(RemoveRedundantTransfersPassTest, MultiMapWithLatterUseTest) {
    sdfg::builder::StructuredSDFGBuilder builder("dot", sdfg::FunctionType_CPU);
    sdfg::analysis::AnalysisManager analysis_manager(builder.subject());

    auto& root = builder.subject().root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("__daisy_offload_A", desc);

    auto& block = builder.add_block(root);
    auto& access_node_in = builder.add_access(block, "__daisy_offload_A");
    auto& access_node_out = builder.add_access(block, "A");

    auto& memcpy_node = builder.add_library_node<cuda::CUDAOffloadingNode>(
        block,
        DebugInfo(),
        symbolic::integer(400),
        symbolic::integer(0),
        memory::DataTransferDirection::D2H,
        memory::BufferLifecycle::NO_CHANGE
    );

    auto& in_type = builder.subject().type("A");
    builder.add_computational_memlet(block, access_node_in, memcpy_node, "_src", {}, in_type);

    auto& out_type = builder.subject().type("A");
    builder.add_computational_memlet(block, memcpy_node, "_dst", access_node_out, {}, out_type);

    auto& block2 = builder.add_block(root);

    auto& access_node_in2 = builder.add_access(block2, "__daisy_offload_A");
    auto& access_node_out2 = builder.add_access(block2, "A");

    auto& memcpy_node2 = builder.add_library_node<cuda::CUDAOffloadingNode>(
        block2,
        DebugInfo(),
        symbolic::integer(400),
        symbolic::integer(0),
        memory::DataTransferDirection::D2H,
        memory::BufferLifecycle::NO_CHANGE
    );

    auto& in_type2 = builder.subject().type("A");
    builder.add_computational_memlet(block2, access_node_in2, memcpy_node2, "_src", {}, in_type2);

    auto& out_type2 = builder.subject().type("A");
    builder.add_computational_memlet(block2, memcpy_node2, "_dst", access_node_out2, {}, out_type2);

    // Add another use of C after the second map
    auto& block3 = builder.add_block(root);
    auto& C3 = builder.add_access(block3, "A");
    auto& tasklet3 = builder.add_tasklet(block3, data_flow::TaskletCode::assign, "_out", {"_in"});
    auto& memlet_c3 = builder.add_computational_memlet(block3, C3, tasklet3, "_in", {symbolic::zero()});

    passes::RemoveRedundantTransfersPass pass;
    EXPECT_TRUE(pass.run_pass(builder, analysis_manager));

    // Check that there is exactly two H2D and one D2H transfer for C
    int h2d_count = 0;
    int d2h_count = 0;
    for (int i = 0; i < root.size(); i++) {
        auto& cf_node = root.at(i).first;
        if (auto* block = dynamic_cast<structured_control_flow::Block*>(&cf_node)) {
            for (auto& node : block->dataflow().nodes()) {
                if (auto* data_transfer = dynamic_cast<memory::OffloadingNode*>(&node)) {
                    if (data_transfer->is_h2d()) {
                        h2d_count++;
                    } else if (data_transfer->is_d2h()) {
                        d2h_count++;
                    }
                }
            }
        }
    }
    EXPECT_EQ(h2d_count, 0);
    EXPECT_EQ(d2h_count, 1);
};
