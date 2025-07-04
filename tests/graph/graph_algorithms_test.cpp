#include <gtest/gtest.h>

#include "sdfg/graph/graph.h"

TEST(GraphAlgorithmsTest, Undirected) {
    sdfg::graph::Graph graph;

    auto u = boost::add_vertex(graph);
    auto v = boost::add_vertex(graph);
    auto edge = boost::add_edge(u, v, graph);

    sdfg::graph::Edge e1, e2;
    bool found;
    boost::tie(e1, found) = boost::edge(u, v, graph);
    EXPECT_TRUE(found);
    boost::tie(e2, found) = boost::edge(v, u, graph);
    EXPECT_FALSE(found);

    const auto& undirected_graph = sdfg::graph::undirected(graph);
    EXPECT_EQ(boost::num_vertices(*std::get<0>(undirected_graph)), 2);
    EXPECT_EQ(boost::num_edges(*std::get<0>(undirected_graph)), 1);

    sdfg::graph::UndirectedEdge e3, e4;
    boost::tie(e3, found) = boost::
        edge(std::get<1>(undirected_graph).at(u), std::get<1>(undirected_graph).at(v), *std::get<0>(undirected_graph));
    EXPECT_TRUE(found);
    boost::tie(e4, found) = boost::
        edge(std::get<1>(undirected_graph).at(v), std::get<1>(undirected_graph).at(u), *std::get<0>(undirected_graph));
    EXPECT_TRUE(found);
}

TEST(GraphAlgorithmsTest, DepthFirstSearch) {
    sdfg::graph::Graph graph;

    auto u = boost::add_vertex(graph);
    auto v1 = boost::add_vertex(graph);
    auto v2 = boost::add_vertex(graph);
    auto z = boost::add_vertex(graph);
    auto edge1 = boost::add_edge(u, v1, graph);
    auto edge2 = boost::add_edge(u, v2, graph);
    auto edge3 = boost::add_edge(v1, z, graph);
    auto edge4 = boost::add_edge(v2, z, graph);

    auto dfs_nodes = sdfg::graph::depth_first_search(graph, u);
    EXPECT_EQ(dfs_nodes.size(), 4);

    auto it = dfs_nodes.begin();
    EXPECT_EQ(*it, u);
    ++it;

    if (*it == v1) {
        EXPECT_EQ(*it, v1);
        ++it;
        EXPECT_EQ(*it, z);
        ++it;
        EXPECT_EQ(*it, v2);
    } else {
        EXPECT_EQ(*it, v2);
        ++it;
        EXPECT_EQ(*it, z);
        ++it;
        EXPECT_EQ(*it, v1);
    }
}

TEST(GraphAlgorithmsTest, Reverse) {
    sdfg::graph::Graph graph;

    auto u = boost::add_vertex(graph);
    auto v = boost::add_vertex(graph);
    auto edge = boost::add_edge(u, v, graph);

    sdfg::graph::Edge e1, e2;
    bool found;
    boost::tie(e1, found) = boost::edge(u, v, graph);
    EXPECT_TRUE(found);
    boost::tie(e2, found) = boost::edge(v, u, graph);
    EXPECT_FALSE(found);

    auto reverse_graph = sdfg::graph::reverse(graph);
    EXPECT_EQ(boost::num_edges(reverse_graph), 1);

    auto [eb, ee] = boost::edges(reverse_graph);
    auto reverse_edges = std::ranges::subrange(eb, ee);
    EXPECT_EQ(boost::source(*eb, reverse_graph), v);
    EXPECT_EQ(boost::target(*eb, reverse_graph), u);
}

TEST(GraphAlgorithmsTest, StronglyConnectedComponents) {
    sdfg::graph::Graph graph;

    auto u = boost::add_vertex(graph);
    auto v = boost::add_vertex(graph);
    auto t = boost::add_vertex(graph);
    boost::add_edge(u, v, graph);
    boost::add_edge(v, u, graph);

    auto ccs = sdfg::graph::strongly_connected_components(graph);
    EXPECT_EQ(ccs.first, 2);
    EXPECT_EQ(ccs.second.at(u), ccs.second.at(v));
    EXPECT_NE(ccs.second.at(u), ccs.second.at(t));
}

TEST(GraphAlgorithmsTest, WeaklyConnectedComponents) {
    sdfg::graph::Graph graph;

    auto u = boost::add_vertex(graph);
    auto v = boost::add_vertex(graph);
    auto t = boost::add_vertex(graph);
    boost::add_edge(u, v, graph);

    auto ccs = sdfg::graph::weakly_connected_components(graph);
    EXPECT_EQ(ccs.first, 2);
    EXPECT_EQ(ccs.second.at(u), ccs.second.at(v));
    EXPECT_NE(ccs.second.at(u), ccs.second.at(t));
}

TEST(GraphAlgorithmsTest, DominatorTree) {
    sdfg::graph::Graph graph;

    auto u = boost::add_vertex(graph);
    auto v = boost::add_vertex(graph);
    auto s = boost::add_vertex(graph);
    auto t = boost::add_vertex(graph);
    auto z = boost::add_vertex(graph);
    boost::add_edge(u, v, graph);
    boost::add_edge(u, s, graph);
    boost::add_edge(v, t, graph);
    boost::add_edge(s, t, graph);
    boost::add_edge(t, z, graph);

    auto dom_tree = sdfg::graph::dominator_tree(graph, u);
    EXPECT_EQ(dom_tree.size(), 5);
    EXPECT_EQ(dom_tree.at(u), boost::graph_traits<sdfg::graph::Graph>::null_vertex());
    EXPECT_EQ(dom_tree.at(s), u);
    EXPECT_EQ(dom_tree.at(v), u);
    EXPECT_EQ(dom_tree.at(t), u);
    EXPECT_EQ(dom_tree.at(z), t);
}

TEST(GraphAlgorithmsTest, PostDominatorTree) {
    sdfg::graph::Graph graph;

    auto u = boost::add_vertex(graph);
    auto v = boost::add_vertex(graph);
    auto s = boost::add_vertex(graph);
    auto t = boost::add_vertex(graph);
    auto z = boost::add_vertex(graph);
    boost::add_edge(u, v, graph);
    boost::add_edge(u, s, graph);
    boost::add_edge(v, t, graph);
    boost::add_edge(s, t, graph);
    boost::add_edge(t, z, graph);

    auto pdom_tree = sdfg::graph::post_dominator_tree(graph, z);
    EXPECT_EQ(pdom_tree.size(), 5);
    EXPECT_EQ(pdom_tree.at(z), boost::graph_traits<sdfg::graph::Graph>::null_vertex());
    EXPECT_EQ(pdom_tree.at(t), z);
    EXPECT_EQ(pdom_tree.at(v), t);
    EXPECT_EQ(pdom_tree.at(s), t);
    EXPECT_EQ(pdom_tree.at(u), t);
}

TEST(GraphAlgorithmsTest, isAcyclic) {
    sdfg::graph::Graph graph;

    auto u = boost::add_vertex(graph);
    auto v = boost::add_vertex(graph);
    auto s = boost::add_vertex(graph);
    auto t = boost::add_vertex(graph);
    boost::add_edge(u, v, graph);
    boost::add_edge(u, s, graph);
    boost::add_edge(v, t, graph);
    boost::add_edge(s, t, graph);

    EXPECT_TRUE(sdfg::graph::is_acyclic(graph));

    boost::add_edge(t, u, graph);
    EXPECT_FALSE(sdfg::graph::is_acyclic(graph));
}

TEST(GraphAlgorithmsTest, TopologicalSort) {
    sdfg::graph::Graph graph;

    auto u = boost::add_vertex(graph);
    auto v = boost::add_vertex(graph);
    auto s = boost::add_vertex(graph);
    auto t = boost::add_vertex(graph);
    auto z = boost::add_vertex(graph);
    boost::add_edge(u, v, graph);
    boost::add_edge(u, s, graph);
    boost::add_edge(v, t, graph);
    boost::add_edge(s, t, graph);
    boost::add_edge(t, z, graph);

    auto topo_nodes = sdfg::graph::topological_sort(graph);
    EXPECT_EQ(topo_nodes.size(), 5);

    auto it = topo_nodes.begin();
    EXPECT_EQ(*it, u);
    ++it;

    if (*it == v) {
        EXPECT_EQ(*it, v);
        ++it;
        EXPECT_EQ(*it, s);
        ++it;
        EXPECT_EQ(*it, t);
        ++it;
        EXPECT_EQ(*it, z);
    } else {
        EXPECT_EQ(*it, s);
        ++it;
        EXPECT_EQ(*it, v);
        ++it;
        EXPECT_EQ(*it, t);
        ++it;
        EXPECT_EQ(*it, z);
    }
}

TEST(GraphAlgorithmsTest, BackEdges) {
    sdfg::graph::Graph graph;

    auto u = boost::add_vertex(graph);
    auto v = boost::add_vertex(graph);
    auto s = boost::add_vertex(graph);
    boost::add_edge(u, v, graph);
    boost::add_edge(v, s, graph);
    auto back_edge = boost::add_edge(s, u, graph);

    auto back_edges = sdfg::graph::back_edges(graph, u);
    EXPECT_EQ(back_edges.size(), 1);
    EXPECT_EQ(*back_edges.begin(), back_edge.first);
}

TEST(GraphAlgorithmsTest, AllSimplePaths) {
    sdfg::graph::Graph graph;

    auto u = boost::add_vertex(graph);
    auto v = boost::add_vertex(graph);
    auto s = boost::add_vertex(graph);
    auto t = boost::add_vertex(graph);
    auto z = boost::add_vertex(graph);
    auto edge1 = boost::add_edge(u, v, graph);
    auto edge2 = boost::add_edge(u, s, graph);
    auto edge3 = boost::add_edge(v, t, graph);
    auto edge4 = boost::add_edge(s, t, graph);
    auto edge5 = boost::add_edge(t, z, graph);

    auto paths = sdfg::graph::all_simple_paths(graph, u, z);
    EXPECT_EQ(paths.size(), 2);

    auto path_1 = paths.begin();
    EXPECT_EQ(path_1->size(), 3);
    auto path_2 = ++paths.begin();
    EXPECT_EQ(path_2->size(), 3);

    auto it = path_1->begin();
    if (*it == edge1.first) {
        EXPECT_EQ(*it, edge1.first);
        ++it;
        EXPECT_EQ(*it, edge3.first);
        ++it;
        EXPECT_EQ(*it, edge5.first);

        auto it2 = path_2->begin();
        EXPECT_EQ(*it2, edge2.first);
        ++it2;
        EXPECT_EQ(*it2, edge4.first);
        ++it2;
        EXPECT_EQ(*it2, edge5.first);
    } else {
        EXPECT_EQ(*it, edge2.first);
        ++it;
        EXPECT_EQ(*it, edge4.first);
        ++it;
        EXPECT_EQ(*it, edge5.first);

        auto it2 = path_2->begin();
        EXPECT_EQ(*it2, edge1.first);
        ++it2;
        EXPECT_EQ(*it2, edge3.first);
        ++it2;
        EXPECT_EQ(*it2, edge5.first);
    }
}
