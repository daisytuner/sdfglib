#include <gtest/gtest.h>

#include "sdfg/builder/sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/call_node.h"
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

TEST(DataflowTest, DiamondGraph) {
    builder::SDFGBuilder builder("diamond_sdfg", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Double);
    builder.add_container("A", desc);
    builder.add_container("B", desc);
    builder.add_container("C", desc);
    builder.add_container("D", desc);

    auto& A = builder.add_access(state, "A");
    auto& B = builder.add_access(state, "B");
    auto& C = builder.add_access(state, "C");
    auto& D = builder.add_access(state, "D");

    // A -> T1 -> B
    auto& T1 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, A, T1, "_in", {});
    builder.add_computational_memlet(state, T1, "_out", B, {});

    // A -> T2 -> C
    auto& T2 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, A, T2, "_in", {});
    builder.add_computational_memlet(state, T2, "_out", C, {});

    // B, C -> T3 -> D
    auto& T3 = builder.add_tasklet(state, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(state, B, T3, "_in1", {});
    builder.add_computational_memlet(state, C, T3, "_in2", {});
    builder.add_computational_memlet(state, T3, "_out", D, {});

    // This should not throw
    std::vector<const data_flow::DataFlowNode*> order;
    EXPECT_NO_THROW({
        auto list = state.dataflow().topological_sort_deterministic();
        order.assign(list.begin(), list.end());
    });

    std::vector<const data_flow::DataFlowNode*> expected = {&A, &T1, &B, &T2, &C, &T3, &D};
    ASSERT_EQ(order.size(), expected.size());
    for (size_t i = 0; i < order.size(); ++i) {
        EXPECT_EQ(order[i], expected[i]) << "Mismatch at index " << i;
    }
}

TEST(DataflowTest, MultiEdgeGraph) {
    builder::SDFGBuilder builder("multi_edge_sdfg", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Double);
    builder.add_container("A", desc);
    builder.add_container("B", desc);

    auto& A = builder.add_access(state, "A");
    auto& B = builder.add_access(state, "B");

    // A -> T1 (two edges)
    auto& T1 = builder.add_tasklet(state, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});

    // Two edges from A to T1
    builder.add_computational_memlet(state, A, T1, "_in1", {});
    builder.add_computational_memlet(state, A, T1, "_in2", {});

    builder.add_computational_memlet(state, T1, "_out", B, {});

    // This should not throw
    std::vector<const data_flow::DataFlowNode*> order;
    EXPECT_NO_THROW({
        auto list = state.dataflow().topological_sort_deterministic();
        order.assign(list.begin(), list.end());
    });

    std::vector<const data_flow::DataFlowNode*> expected = {&A, &T1, &B};
    ASSERT_EQ(order.size(), expected.size());
    for (size_t i = 0; i < order.size(); ++i) {
        EXPECT_EQ(order[i], expected[i]) << "Mismatch at index " << i;
    }
}
TEST(DataflowTest, CrossedDependencies) {
    builder::SDFGBuilder builder("crossed_sdfg", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Double);
    builder.add_container("A", desc);
    builder.add_container("B", desc);
    builder.add_container("S1", desc);
    builder.add_container("S2", desc);

    auto& A = builder.add_access(state, "A");
    auto& B = builder.add_access(state, "B");
    auto& S1 = builder.add_access(state, "S1");
    auto& S2 = builder.add_access(state, "S2");

    // CallNode T1 (acts as A source logic)
    // Outputs: _out1 (Primary) -> S1, _out2 (Secondary) -> S2
    auto& T1 = builder.add_library_node<data_flow::CallNode>(
        state, DebugInfo(), "func1", std::vector<std::string>{"_out1", "_out2"}, std::vector<std::string>{"_in"}
    );
    builder.add_computational_memlet(state, A, T1, "_in", {}, desc);
    builder.add_computational_memlet(state, T1, "_out1", S1, {}, desc);
    builder.add_computational_memlet(state, T1, "_out2", S2, {}, desc);

    // CallNode T2 (acts as B source logic)
    // Outputs: _out1 (Primary) -> S2, _out2 (Secondary) -> S1
    auto& T2 = builder.add_library_node<data_flow::CallNode>(
        state, DebugInfo(), "func2", std::vector<std::string>{"_out1", "_out2"}, std::vector<std::string>{"_in"}
    );
    builder.add_computational_memlet(state, B, T2, "_in", {}, desc);
    builder.add_computational_memlet(state, T2, "_out1", S2, {}, desc);
    builder.add_computational_memlet(state, T2, "_out2", S1, {}, desc);

    // This is expected to throw boost::not_a_dag with the current implementation
    // because both S1 and S2 will be "blocked" by secondary edges.
    std::vector<const data_flow::DataFlowNode*> order;
    EXPECT_NO_THROW({
        auto list = state.dataflow().topological_sort_deterministic();
        order.assign(list.begin(), list.end());
    });

    std::vector<const data_flow::DataFlowNode*> expected = {&A, &T1, &B, &T2, &S1, &S2};
    ASSERT_EQ(order.size(), expected.size());
    for (size_t i = 0; i < order.size(); ++i) {
        EXPECT_EQ(order[i], expected[i]) << "Mismatch at index " << i;
    }
}
TEST(DataflowTest, FanOutGraph) {
    builder::SDFGBuilder builder("fan_out_sdfg", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Double);
    builder.add_container("A", desc);
    builder.add_container("B", desc);
    builder.add_container("C", desc);
    builder.add_container("S1", desc);
    builder.add_container("S2", desc);

    auto& A = builder.add_access(state, "A");
    auto& B = builder.add_access(state, "B");
    auto& C = builder.add_access(state, "C");
    auto& S1 = builder.add_access(state, "S1");
    auto& S2 = builder.add_access(state, "S2");

    // A -> B -> S1
    auto& T1 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, A, T1, "_in", {});
    builder.add_computational_memlet(state, T1, "_out", B, {});

    auto& T2 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, B, T2, "_in", {});
    builder.add_computational_memlet(state, T2, "_out", S1, {});

    // A -> C -> S2
    auto& T3 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, A, T3, "_in", {});
    builder.add_computational_memlet(state, T3, "_out", C, {});

    auto& T4 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, C, T4, "_in", {});
    builder.add_computational_memlet(state, T4, "_out", S2, {});

    // This should not throw
    std::vector<const data_flow::DataFlowNode*> order;
    EXPECT_NO_THROW({
        auto list = state.dataflow().topological_sort_deterministic();
        order.assign(list.begin(), list.end());
        // Verify no duplicates
        std::unordered_set<const data_flow::DataFlowNode*> unique_nodes;
        for (auto* node : order) {
            EXPECT_TRUE(unique_nodes.find(node) == unique_nodes.end()) << "Duplicate node in topological sort";
            unique_nodes.insert(node);
        }
    });

    std::vector<const data_flow::DataFlowNode*> expected = {&A, &T1, &B, &T2, &S1, &T3, &C, &T4, &S2};
    ASSERT_EQ(order.size(), expected.size());
    for (size_t i = 0; i < order.size(); ++i) {
        EXPECT_EQ(order[i], expected[i]) << "Mismatch at index " << i;
    }
}

TEST(DataflowTest, ComplexMultiInput) {
    builder::SDFGBuilder builder("complex_multi_input", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Double);
    builder.add_container("A", desc);
    builder.add_container("B", desc);
    builder.add_container("C", desc);

    auto& A = builder.add_access(state, "A");
    auto& B = builder.add_access(state, "B");
    auto& C = builder.add_access(state, "C");

    // A -> T1 -> B
    auto& T1 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, A, T1, "_in", {});
    builder.add_computational_memlet(state, T1, "_out", B, {});

    // A -> T2 -> B
    auto& T2 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, A, T2, "_in", {});
    builder.add_computational_memlet(state, T2, "_out", B, {});

    // B -> T3 -> C
    auto& T3 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, B, T3, "_in", {});
    builder.add_computational_memlet(state, T3, "_out", C, {});

    std::vector<const data_flow::DataFlowNode*> order;
    EXPECT_NO_THROW({
        auto list = state.dataflow().topological_sort_deterministic();
        order.assign(list.begin(), list.end());
    });

    std::vector<const data_flow::DataFlowNode*> expected = {&A, &T1, &T2, &B, &T3, &C};
    ASSERT_EQ(order.size(), expected.size());
    for (size_t i = 0; i < order.size(); ++i) {
        EXPECT_EQ(order[i], expected[i]) << "Mismatch at index " << i;
    }
}

TEST(DataflowTest, ComplexBranching) {
    builder::SDFGBuilder builder("complex_branching", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Double);
    builder.add_container("A", desc);
    builder.add_container("B", desc);
    builder.add_container("C", desc);
    builder.add_container("D", desc);
    builder.add_container("E", desc);

    auto& A = builder.add_access(state, "A");
    auto& B = builder.add_access(state, "B");
    auto& C = builder.add_access(state, "C");
    auto& D = builder.add_access(state, "D");
    auto& E = builder.add_access(state, "E");

    // A -> T1 -> B
    auto& T1 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, A, T1, "_in", {});
    builder.add_computational_memlet(state, T1, "_out", B, {});

    // A -> T2 -> C
    auto& T2 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, A, T2, "_in", {});
    builder.add_computational_memlet(state, T2, "_out", C, {});

    // B, C -> T3 -> D
    auto& T3 = builder.add_tasklet(state, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(state, B, T3, "_in1", {});
    builder.add_computational_memlet(state, C, T3, "_in2", {});
    builder.add_computational_memlet(state, T3, "_out", D, {});

    // D -> T4 -> E
    auto& T4 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, D, T4, "_in", {});
    builder.add_computational_memlet(state, T4, "_out", E, {});

    std::vector<const data_flow::DataFlowNode*> order;
    EXPECT_NO_THROW({
        auto list = state.dataflow().topological_sort_deterministic();
        order.assign(list.begin(), list.end());
    });

    std::vector<const data_flow::DataFlowNode*> expected = {&A, &T1, &B, &T2, &C, &T3, &D, &T4, &E};
    ASSERT_EQ(order.size(), expected.size());
    for (size_t i = 0; i < order.size(); ++i) {
        EXPECT_EQ(order[i], expected[i]) << "Mismatch at index " << i;
    }
}

TEST(DataflowTest, TriangleWithCrossEdge) {
    builder::SDFGBuilder builder("triangle_cross", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Double);
    builder.add_container("A", desc);
    builder.add_container("B", desc);
    builder.add_container("C", desc);
    builder.add_container("D", desc);

    auto& A = builder.add_access(state, "A");
    auto& B = builder.add_access(state, "B");
    auto& C = builder.add_access(state, "C");
    auto& D = builder.add_access(state, "D");

    // A -> T1 -> B
    auto& T1 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, A, T1, "_in", {});
    builder.add_computational_memlet(state, T1, "_out", B, {});

    // A -> T2 -> C
    auto& T2 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, A, T2, "_in", {});
    builder.add_computational_memlet(state, T2, "_out", C, {});

    // B -> T3 -> C (Cross edge)
    auto& T3 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, B, T3, "_in", {});
    builder.add_computational_memlet(state, T3, "_out", C, {});

    // B -> T4 -> D
    auto& T4 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, B, T4, "_in", {});
    builder.add_computational_memlet(state, T4, "_out", D, {});

    // C -> T5 -> D
    auto& T5 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, C, T5, "_in", {});
    builder.add_computational_memlet(state, T5, "_out", D, {});

    std::vector<const data_flow::DataFlowNode*> order;
    EXPECT_NO_THROW({
        auto list = state.dataflow().topological_sort_deterministic();
        order.assign(list.begin(), list.end());
    });

    std::vector<const data_flow::DataFlowNode*> expected = {&A, &T1, &B, &T4, &T3, &T2, &C, &T5, &D};
    ASSERT_EQ(order.size(), expected.size());
    for (size_t i = 0; i < order.size(); ++i) {
        EXPECT_EQ(order[i], expected[i]) << "Mismatch at index " << i;
    }
}
