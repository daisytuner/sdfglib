#include <gtest/gtest.h>

#include "sdfg/builder/sdfg_builder.h"
using namespace sdfg;

TEST(SDFGTest, Metadata) {
    builder::SDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    sdfg.add_metadata("key", "value");

    EXPECT_EQ(sdfg.metadata("key"), "value");

    sdfg.remove_metadata("key");
    EXPECT_THROW(sdfg.metadata("key"), std::out_of_range);
}

TEST(SDFGTest, InAndOutDegree) {
    builder::SDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& state_1 = builder.add_state(true);
    auto& state_2 = builder.add_state();
    auto& state_3 = builder.add_state();
    auto& state_4 = builder.add_state();
    auto& edge_1 = builder.add_edge(state_1, state_2);
    auto& edge_2 = builder.add_edge(state_1, state_3);
    auto& edge_3 = builder.add_edge(state_2, state_4);
    auto& edge_4 = builder.add_edge(state_3, state_4);

    auto& sdfg = builder.subject();

    EXPECT_EQ(sdfg.in_degree(state_1), 0);
    EXPECT_EQ(sdfg.out_degree(state_1), 2);
    EXPECT_EQ(sdfg.in_degree(state_2), 1);
    EXPECT_EQ(sdfg.out_degree(state_2), 1);
    EXPECT_EQ(sdfg.in_degree(state_3), 1);
    EXPECT_EQ(sdfg.out_degree(state_3), 1);
    EXPECT_EQ(sdfg.in_degree(state_4), 2);
    EXPECT_EQ(sdfg.out_degree(state_4), 0);
}

TEST(SDFGTest, IsAdjacent) {
    builder::SDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& state_1 = builder.add_state(true);
    auto& state_2 = builder.add_state();
    auto& state_3 = builder.add_state();
    auto& state_4 = builder.add_state();
    auto& edge_1 = builder.add_edge(state_1, state_2);
    auto& edge_2 = builder.add_edge(state_1, state_3);
    auto& edge_3 = builder.add_edge(state_2, state_4);
    auto& edge_4 = builder.add_edge(state_3, state_4);

    auto& sdfg = builder.subject();

    EXPECT_TRUE(sdfg.is_adjacent(state_1, state_2));
    EXPECT_TRUE(sdfg.is_adjacent(state_1, state_3));
    EXPECT_TRUE(sdfg.is_adjacent(state_2, state_4));
    EXPECT_TRUE(sdfg.is_adjacent(state_3, state_4));
    EXPECT_FALSE(sdfg.is_adjacent(state_1, state_4));
    EXPECT_FALSE(sdfg.is_adjacent(state_2, state_3));
}

TEST(SDFGTest, Edge) {
    builder::SDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& state_1 = builder.add_state(true);
    auto& state_2 = builder.add_state();
    auto& state_3 = builder.add_state();
    auto& state_4 = builder.add_state();
    auto& edge_1 = builder.add_edge(state_1, state_2);
    auto& edge_2 = builder.add_edge(state_1, state_3);
    auto& edge_3 = builder.add_edge(state_2, state_4);
    auto& edge_4 = builder.add_edge(state_3, state_4);

    auto& sdfg = builder.subject();

    EXPECT_EQ(&sdfg.edge(state_1, state_2), &edge_1);
    EXPECT_EQ(&sdfg.edge(state_1, state_3), &edge_2);
    EXPECT_EQ(&sdfg.edge(state_2, state_4), &edge_3);
    EXPECT_EQ(&sdfg.edge(state_3, state_4), &edge_4);
}

TEST(SDFGTest, DominatorTree) {
    builder::SDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& state_1 = builder.add_state(true);
    auto& state_2 = builder.add_state();
    auto& state_3 = builder.add_state();
    auto& state_4 = builder.add_state();
    auto& state_5 = builder.add_state();
    auto& edge_1 = builder.add_edge(state_1, state_2);
    auto& edge_2 = builder.add_edge(state_1, state_3);
    auto& edge_3 = builder.add_edge(state_2, state_4);
    auto& edge_4 = builder.add_edge(state_3, state_4);
    auto& edge_5 = builder.add_edge(state_4, state_5);

    auto& sdfg = builder.subject();

    auto dominator_tree = sdfg.dominator_tree();
    EXPECT_EQ(dominator_tree.size(), 5);

    EXPECT_EQ(dominator_tree.at(&state_1), nullptr);
    EXPECT_EQ(dominator_tree.at(&state_2), &state_1);
    EXPECT_EQ(dominator_tree.at(&state_3), &state_1);
    EXPECT_EQ(dominator_tree.at(&state_4), &state_1);
    EXPECT_EQ(dominator_tree.at(&state_5), &state_4);
}

TEST(SDFGTest, PostDominatorTree) {
    builder::SDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& state_1 = builder.add_state(true);
    auto& state_2 = builder.add_state();
    auto& state_3 = builder.add_state();
    auto& state_4 = builder.add_state();
    auto& state_5 = builder.add_state();
    auto& edge_1 = builder.add_edge(state_1, state_2);
    auto& edge_2 = builder.add_edge(state_1, state_3);
    auto& edge_3 = builder.add_edge(state_2, state_4);
    auto& edge_4 = builder.add_edge(state_3, state_4);
    auto& edge_5 = builder.add_edge(state_4, state_5);

    auto& sdfg = builder.subject();

    auto post_dominator_tree = sdfg.post_dominator_tree();
    EXPECT_EQ(post_dominator_tree.size(), 5);

    EXPECT_EQ(post_dominator_tree.at(&state_1), &state_4);
    EXPECT_EQ(post_dominator_tree.at(&state_2), &state_4);
    EXPECT_EQ(post_dominator_tree.at(&state_3), &state_4);
    EXPECT_EQ(post_dominator_tree.at(&state_4), &state_5);
    EXPECT_EQ(post_dominator_tree.at(&state_5), nullptr);
}

static std::tuple<int, int, int> count_exits(const sdfg::SDFG& sdfg, const sdfg::control_flow::State& header) {
    auto& cls = const_cast<sdfg::SDFG&>(sdfg).loop_exits();
    auto it = cls.find(&header);
    if (it == cls.end()) return {0, 0, 0};
    int cont = 0, brk = 0, ret = 0;
    for (auto& ex : it->second) {
        switch (ex.kind) {
            case sdfg::analysis::LoopExitKind::Continue:
                cont++;
                break;
            case sdfg::analysis::LoopExitKind::Break:
                brk++;
                break;
            case sdfg::analysis::LoopExitKind::Return:
                ret++;
                break;
        }
    }
    return {cont, brk, ret};
}

TEST(SDFGTest, LoopExits_ContinueAndBreak) {
    sdfg::builder::SDFGBuilder b("continue_and_break", sdfg::FunctionType_CPU);
    auto& entry = b.add_state(true);
    auto& h = b.add_state();
    auto& latch = b.add_state();
    auto& exit = b.add_state();
    b.add_edge(entry, h);
    b.add_edge(h, latch);
    b.add_edge(latch, h); // continue
    b.add_edge(h, exit);
    auto& sdfg = b.subject();
    ASSERT_EQ(sdfg.natural_loops().size(), 1);
    auto [c, bk, rt] = count_exits(sdfg, h);
    EXPECT_EQ(c, 1);
    EXPECT_EQ(bk, 1);
    EXPECT_EQ(rt, 0);
}

TEST(SDFGTest, LoopExits_MultiExit) {
    sdfg::builder::SDFGBuilder b("multi_exit", sdfg::FunctionType_CPU);
    auto& entry = b.add_state(true);
    auto& h = b.add_state();
    auto& body = b.add_state();
    auto& ret = b.add_return_state("");
    auto& brk1 = b.add_state();
    auto& brk2 = b.add_state();
    b.add_edge(entry, h);
    b.add_edge(h, body);
    b.add_edge(body, h); // continue
    b.add_edge(h, ret); // return
    b.add_edge(h, brk1); // break
    b.add_edge(body, brk2); // break from body
    auto& sdfg = b.subject();
    ASSERT_EQ(sdfg.natural_loops().size(), 1);
    auto [c, bk, rt] = count_exits(sdfg, h);
    EXPECT_EQ(c, 1);
    EXPECT_EQ(rt, 1);
    EXPECT_EQ(bk, 2);
}

TEST(SDFGTest, LoopExits_NestedLoops) {
    sdfg::builder::SDFGBuilder b("nested", sdfg::FunctionType_CPU);
    auto& entry = b.add_state(true);
    auto& h1 = b.add_state();
    auto& inner = b.add_state();
    auto& h2 = b.add_state();
    auto& l2 = b.add_state();
    auto& exit = b.add_state();
    b.add_edge(entry, h1);
    b.add_edge(h1, inner);
    b.add_edge(inner, h2);
    b.add_edge(h2, l2);
    b.add_edge(l2, h2); // inner continue
    b.add_edge(h2, h1); // inner break to outer
    b.add_edge(inner, h1); // outer latch
    b.add_edge(h1, exit);
    auto& sdfg = b.subject();
    auto loops = sdfg.natural_loops();
    ASSERT_EQ(loops.size(), 2);
    auto [c1, bk1, rt1] = count_exits(sdfg, h1);
    EXPECT_EQ(c1, 1);
    EXPECT_EQ(bk1, 1);
    EXPECT_EQ(rt1, 0);
    auto [c2, bk2, rt2] = count_exits(sdfg, h2);
    EXPECT_EQ(c2, 1);
    EXPECT_EQ(bk2, 1);
    EXPECT_EQ(rt2, 0);
}

TEST(SDFGTest, LoopExits_MultiExits2) {
    sdfg::builder::SDFGBuilder builder("test", sdfg::FunctionType_CPU);
    auto& sdfg = builder.subject();

    auto& entry = builder.add_state(true);
    auto& header = builder.add_state();
    auto& body = builder.add_state();
    auto& ret = builder.add_return_state("");
    auto& brk_target = builder.add_state();

    builder.add_edge(entry, header);
    builder.add_edge(header, body);
    builder.add_edge(body, header); // continue
    builder.add_edge(header, ret); // return path
    builder.add_edge(header, brk_target); // break path
    builder.add_edge(body, brk_target); // extra break

    auto loops = sdfg.natural_loops();
    ASSERT_EQ(loops.size(), 1);

    auto& classification = sdfg.loop_exits();
    auto it = classification.find(&header);
    ASSERT_TRUE(it != classification.end());

    int continue_count = 0, break_count = 0, return_count = 0;
    for (const auto& ex : it->second) {
        switch (ex.kind) {
            case sdfg::analysis::LoopExitKind::Continue:
                continue_count++;
                break;
            case sdfg::analysis::LoopExitKind::Break:
                break_count++;
                break;
            case sdfg::analysis::LoopExitKind::Return:
                return_count++;
                break;
        }
    }
    EXPECT_EQ(continue_count, 1);
    EXPECT_EQ(break_count, 2);
    EXPECT_EQ(return_count, 1);
}
