#include <gtest/gtest.h>

#include "sdfg/builder/sdfg_builder.h"
using namespace sdfg;

TEST(SDFGTest, InAndOutDegree) {
    builder::SDFGBuilder builder("sdfg_1", FunctionType::CPU);

    auto& state_1 = builder.add_state(true);
    auto& state_2 = builder.add_state();
    auto& state_3 = builder.add_state();
    auto& state_4 = builder.add_state();
    auto& edge_1 = builder.add_edge(state_1, state_2);
    auto& edge_2 = builder.add_edge(state_1, state_3);
    auto& edge_3 = builder.add_edge(state_2, state_4);
    auto& edge_4 = builder.add_edge(state_3, state_4);

    auto sdfg = builder.move();

    EXPECT_EQ(sdfg->in_degree(state_1), 0);
    EXPECT_EQ(sdfg->out_degree(state_1), 2);
    EXPECT_EQ(sdfg->in_degree(state_2), 1);
    EXPECT_EQ(sdfg->out_degree(state_2), 1);
    EXPECT_EQ(sdfg->in_degree(state_3), 1);
    EXPECT_EQ(sdfg->out_degree(state_3), 1);
    EXPECT_EQ(sdfg->in_degree(state_4), 2);
    EXPECT_EQ(sdfg->out_degree(state_4), 0);
}

TEST(SDFGTest, IsAdjacent) {
    builder::SDFGBuilder builder("sdfg_1", FunctionType::CPU);

    auto& state_1 = builder.add_state(true);
    auto& state_2 = builder.add_state();
    auto& state_3 = builder.add_state();
    auto& state_4 = builder.add_state();
    auto& edge_1 = builder.add_edge(state_1, state_2);
    auto& edge_2 = builder.add_edge(state_1, state_3);
    auto& edge_3 = builder.add_edge(state_2, state_4);
    auto& edge_4 = builder.add_edge(state_3, state_4);

    auto sdfg = builder.move();

    EXPECT_TRUE(sdfg->is_adjacent(state_1, state_2));
    EXPECT_TRUE(sdfg->is_adjacent(state_1, state_3));
    EXPECT_TRUE(sdfg->is_adjacent(state_2, state_4));
    EXPECT_TRUE(sdfg->is_adjacent(state_3, state_4));
    EXPECT_FALSE(sdfg->is_adjacent(state_1, state_4));
    EXPECT_FALSE(sdfg->is_adjacent(state_2, state_3));
}

TEST(SDFGTest, Edge) {
    builder::SDFGBuilder builder("sdfg_1", FunctionType::CPU);

    auto& state_1 = builder.add_state(true);
    auto& state_2 = builder.add_state();
    auto& state_3 = builder.add_state();
    auto& state_4 = builder.add_state();
    auto& edge_1 = builder.add_edge(state_1, state_2);
    auto& edge_2 = builder.add_edge(state_1, state_3);
    auto& edge_3 = builder.add_edge(state_2, state_4);
    auto& edge_4 = builder.add_edge(state_3, state_4);

    auto sdfg = builder.move();

    EXPECT_EQ(&sdfg->edge(state_1, state_2), &edge_1);
    EXPECT_EQ(&sdfg->edge(state_1, state_3), &edge_2);
    EXPECT_EQ(&sdfg->edge(state_2, state_4), &edge_3);
    EXPECT_EQ(&sdfg->edge(state_3, state_4), &edge_4);
}

TEST(SDFGTest, DominatorTree) {
    builder::SDFGBuilder builder("sdfg_1", FunctionType::CPU);

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

    auto sdfg = builder.move();

    auto dominator_tree = sdfg->dominator_tree();
    EXPECT_EQ(dominator_tree.size(), 5);

    EXPECT_EQ(dominator_tree.at(&state_1), nullptr);
    EXPECT_EQ(dominator_tree.at(&state_2), &state_1);
    EXPECT_EQ(dominator_tree.at(&state_3), &state_1);
    EXPECT_EQ(dominator_tree.at(&state_4), &state_1);
    EXPECT_EQ(dominator_tree.at(&state_5), &state_4);
}

TEST(SDFGTest, PostDominatorTree) {
    builder::SDFGBuilder builder("sdfg_1", FunctionType::CPU);

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

    auto sdfg = builder.move();

    auto post_dominator_tree = sdfg->post_dominator_tree();
    EXPECT_EQ(post_dominator_tree.size(), 5);

    EXPECT_EQ(post_dominator_tree.at(&state_1), &state_4);
    EXPECT_EQ(post_dominator_tree.at(&state_2), &state_4);
    EXPECT_EQ(post_dominator_tree.at(&state_3), &state_4);
    EXPECT_EQ(post_dominator_tree.at(&state_4), &state_5);
    EXPECT_EQ(post_dominator_tree.at(&state_5), nullptr);
}

TEST(SDFGTest, AllSimplePaths) {
    builder::SDFGBuilder builder("sdfg_1", FunctionType::CPU);

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

    auto sdfg = builder.move();

    auto paths = sdfg->all_simple_paths(state_1, state_5);
    EXPECT_EQ(paths.size(), 2);

    auto path_1 = paths.begin();
    EXPECT_EQ(path_1->size(), 3);
    auto path_2 = ++paths.begin();
    EXPECT_EQ(path_2->size(), 3);

    auto it = path_1->begin();
    if (*it == &edge_1) {
        EXPECT_EQ(*it, &edge_1);
        ++it;
        EXPECT_EQ(*it, &edge_3);
        ++it;
        EXPECT_EQ(*it, &edge_5);

        auto it_2 = path_2->begin();
        EXPECT_EQ(*it_2, &edge_2);
        ++it_2;
        EXPECT_EQ(*it_2, &edge_4);
        ++it_2;
        EXPECT_EQ(*it_2, &edge_5);
    } else {
        EXPECT_EQ(*it, &edge_2);
        ++it;
        EXPECT_EQ(*it, &edge_4);
        ++it;
        EXPECT_EQ(*it, &edge_5);

        auto it_2 = path_2->begin();
        EXPECT_EQ(*it_2, &edge_1);
        ++it_2;
        EXPECT_EQ(*it_2, &edge_3);
        ++it_2;
        EXPECT_EQ(*it_2, &edge_5);
    }
}

TEST(SDFGTest, Metadata) {
    builder::SDFGBuilder builder("sdfg_1", FunctionType::CPU);

    auto sdfg = builder.move();
    sdfg->add_metadata("key", "value");

    EXPECT_EQ(sdfg->metadata("key"), "value");

    sdfg->remove_metadata("key");
    EXPECT_THROW(sdfg->metadata("key"), std::out_of_range);
}
