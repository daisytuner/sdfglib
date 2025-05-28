#include <gtest/gtest.h>

#include "sdfg/builder/sdfg_builder.h"
using namespace sdfg;

TEST(DataflowTest, TopologicalSort) {
    builder::SDFGBuilder builder("sdfg_1");

    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Double);
    builder.add_container("scalar_1", desc);
    builder.add_container("scalar_2", desc);

    auto& access_node_1 = builder.add_access(state, "scalar_1");
    auto& access_node_2 = builder.add_access(state, "scalar_2");
    auto& access_node_3 = builder.add_access(state, "scalar_1");
    auto& tasklet_1 = builder.add_tasklet(state, data_flow::TaskletCode::add, {"_out", desc},
                                          {{"_in", desc}, {"1", desc}});
    auto& tasklet_2 = builder.add_tasklet(state, data_flow::TaskletCode::add, {"_out", desc},
                                          {{"_in", desc}, {"2", desc}});

    builder.add_memlet(state, access_node_1, "void", tasklet_1, "_in", {});
    builder.add_memlet(state, tasklet_1, "_out", access_node_2, "void", {});
    builder.add_memlet(state, access_node_2, "void", tasklet_2, "_in", {});
    builder.add_memlet(state, tasklet_2, "_out", access_node_3, "void", {});

    auto sdfg = builder.move();

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
