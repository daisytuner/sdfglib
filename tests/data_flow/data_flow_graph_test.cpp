#include <gtest/gtest.h>

#include "sdfg/builder/sdfg_builder.h"
#include "sdfg/data_flow/data_flow_node.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/function.h"
#include "sdfg/types/type.h"
using namespace sdfg;

TEST(DataflowTest, TopologicalSort) {
    builder::SDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Double);
    builder.add_container("scalar_1", desc);
    builder.add_container("scalar_2", desc);

    auto& access_node_1 = builder.add_access(state, "scalar_1");
    auto& access_node_2 = builder.add_access(state, "scalar_2");
    auto& access_node_3 = builder.add_access(state, "scalar_1");
    auto& tasklet_1 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    auto& tasklet_2 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});

    builder.add_computational_memlet(state, access_node_1, tasklet_1, "_in", {});
    builder.add_computational_memlet(state, tasklet_1, "_out", access_node_2, {});
    builder.add_computational_memlet(state, access_node_2, tasklet_2, "_in", {});
    builder.add_computational_memlet(state, tasklet_2, "_out", access_node_3, {});

    int i = 0;
    auto order = state.dataflow().topological_sort();
    for (auto& node : order) {
        if (i == 0) {
            EXPECT_EQ(node, &access_node_1);
        } else if (i == 1) {
            EXPECT_EQ(node, &tasklet_1);
        } else if (i == 2) {
            EXPECT_EQ(node, &access_node_2);
        } else if (i == 3) {
            EXPECT_EQ(node, &tasklet_2);
        } else if (i == 4) {
            EXPECT_EQ(node, &access_node_3);
        } else {
            EXPECT_TRUE(false);
        }

        i++;
    }
    EXPECT_EQ(i, 5);
}

TEST(DataflowTest, DeterministicTopologicalSort) {
    builder::SDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Double);
    builder.add_container("A", desc);
    builder.add_container("B", desc);
    builder.add_container("C", desc);
    builder.add_container("D", desc);
    builder.add_container("E", desc);
    builder.add_container("F", desc);
    builder.add_container("G", desc);
    builder.add_container("H", desc);
    builder.add_container("I", desc);
    builder.add_container("J", desc);
    builder.add_container("K", desc);
    builder.add_container("L", desc);
    builder.add_container("M", desc);
    builder.add_container("N", desc);

    auto& state = builder.add_state();

    auto& A = builder.add_access(state, "A");
    auto& B = builder.add_access(state, "B");
    auto& C = builder.add_access(state, "C");
    auto& D = builder.add_access(state, "D");
    auto& E = builder.add_access(state, "E");
    auto& F = builder.add_access(state, "F");
    auto& G = builder.add_access(state, "G");
    auto& H = builder.add_access(state, "H");
    auto& I = builder.add_access(state, "I");
    auto& J = builder.add_access(state, "J");
    auto& K = builder.add_access(state, "K");
    auto& L = builder.add_access(state, "L");
    auto& M = builder.add_access(state, "M");
    auto& N = builder.add_access(state, "N");

    auto& tasklet1 = builder.add_tasklet(state, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(state, A, tasklet1, "_in1", {});
    builder.add_computational_memlet(state, B, tasklet1, "_in2", {});
    builder.add_computational_memlet(state, tasklet1, "_out", C, {});

    auto& tasklet2 = builder.add_tasklet(state, data_flow::TaskletCode::fp_sub, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(state, F, tasklet2, "_in2", {});
    builder.add_computational_memlet(state, E, tasklet2, "_in1", {});
    builder.add_computational_memlet(state, tasklet2, "_out", G, {});

    auto& tasklet3 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, C, tasklet3, "_in", {});
    builder.add_computational_memlet(state, tasklet3, "_out", D, {});

    auto& tasklet4 = builder.add_tasklet(state, data_flow::TaskletCode::fp_add, "_out", {"_in2", "_in1"});
    builder.add_computational_memlet(state, J, tasklet4, "_in2", {});
    builder.add_computational_memlet(state, K, tasklet4, "_in1", {});
    builder.add_computational_memlet(state, tasklet4, "_out", L, {});

    auto& tasklet5 = builder.add_tasklet(state, data_flow::TaskletCode::fp_div, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(state, D, tasklet5, "_in1", {});
    builder.add_computational_memlet(state, G, tasklet5, "_in2", {});
    builder.add_computational_memlet(state, tasklet5, "_out", H, {});

    auto& tasklet6 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, G, tasklet6, "_in", {});
    builder.add_computational_memlet(state, tasklet6, "_out", I, {});

    auto& tasklet7 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, K, tasklet7, "_in", {});
    builder.add_computational_memlet(state, tasklet7, "_out", M, {});

    auto& tasklet8 = builder.add_tasklet(state, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(state, L, tasklet8, "_in1", {});
    builder.add_computational_memlet(state, M, tasklet8, "_in2", {});
    builder.add_computational_memlet(state, tasklet8, "_out", N, {});

    // Expected output:
    // A, B, tasklet1, C, tasklet3, D, E, F, tasklet2, G, tasklet5, H, tasklet6, I,
    // J, K, tasklet4, L, tasklet7, M, tasklet8, N
    auto order = state.dataflow().topological_sort();
    const std::vector<data_flow::DataFlowNode*> expected = {&A,        &B, &tasklet1, &C, &tasklet3, &D, &E, &F,
                                                            &tasklet2, &G, &tasklet5, &H, &tasklet6, &I, &J, &K,
                                                            &tasklet4, &L, &tasklet7, &M, &tasklet8, &N};
    ASSERT_EQ(order.size(), expected.size());
    int i = 0;
    for (auto* node : order) {
        EXPECT_EQ(node, expected.at(i));
        i++;
    }
}
