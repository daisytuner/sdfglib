#include <gtest/gtest.h>

#include "sdfg/graph/graph.h"

TEST(GraphTest, Undirected) {
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

TEST(GraphTest, DepthFirstSearch) {
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

TEST(GraphTest, Reverse) {
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

TEST(GraphTest, isAcyclic) {
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

TEST(GraphTest, TopologicalSort) {
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

TEST(GraphTest, BackEdges) {
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

TEST(GraphTest, StronglyConnectedComponents) {
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

TEST(GraphTest, WeaklyConnectedComponents) {
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

TEST(GraphTest, DominatorTree) {
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

TEST(GraphTest, PostDominatorTree) {
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

    auto pdom_tree = sdfg::graph::post_dominator_tree(graph);
    EXPECT_EQ(pdom_tree.size(), 5);
    EXPECT_EQ(pdom_tree.at(z), boost::graph_traits<sdfg::graph::Graph>::null_vertex());
    EXPECT_EQ(pdom_tree.at(t), z);
    EXPECT_EQ(pdom_tree.at(v), t);
    EXPECT_EQ(pdom_tree.at(s), t);
    EXPECT_EQ(pdom_tree.at(u), t);
}

TEST(GraphTest, DominanceFrontier_SimpleDiamond) {
    sdfg::graph::Graph graph;
    auto entry = boost::add_vertex(graph);
    auto a = boost::add_vertex(graph);
    auto b = boost::add_vertex(graph);
    auto c = boost::add_vertex(graph);
    auto exit = boost::add_vertex(graph);
    boost::add_edge(entry, a, graph);
    boost::add_edge(a, b, graph);
    boost::add_edge(a, c, graph);
    boost::add_edge(c, exit, graph);
    boost::add_edge(b, exit, graph);

    auto idom = sdfg::graph::dominator_tree(graph, entry);
    auto df = sdfg::graph::dominance_frontiers(graph, idom);

    // Expect all nodes present
    EXPECT_EQ(df.size(), idom.size());

    // c frontier contains exit
    ASSERT_TRUE(df.find(c) != df.end());
    EXPECT_TRUE(df.at(c).find(exit) != df.at(c).end());

    // b frontier contains exit
    ASSERT_TRUE(df.find(b) != df.end());
    EXPECT_TRUE(df.at(b).find(exit) != df.at(b).end());

    // a's frontier empty (both successors b,c have idom a)
    EXPECT_TRUE(df.at(a).empty());
    // entry frontier empty
    EXPECT_TRUE(df.at(entry).empty());
    // b frontier not empty (contains exit)
    EXPECT_FALSE(df.at(b).empty());
}

TEST(GraphTest, DominanceFrontier_ExtendedBranch) {
    sdfg::graph::Graph graph;
    auto entry = boost::add_vertex(graph);
    auto a = boost::add_vertex(graph);
    auto b = boost::add_vertex(graph);
    auto c = boost::add_vertex(graph);
    auto d = boost::add_vertex(graph);
    auto exit = boost::add_vertex(graph);
    boost::add_edge(entry, a, graph);
    boost::add_edge(a, b, graph);
    boost::add_edge(a, c, graph);
    boost::add_edge(c, d, graph);
    boost::add_edge(b, exit, graph);
    boost::add_edge(d, exit, graph);

    auto idom = sdfg::graph::dominator_tree(graph, entry);
    auto df = sdfg::graph::dominance_frontiers(graph, idom);

    // c should have frontier containing exit (successor via d path); d dominated by c so excluded
    ASSERT_TRUE(df.find(c) != df.end());
    EXPECT_TRUE(df.at(c).find(exit) != df.at(c).end());

    // a frontier empty again (direct successors dominated)
    EXPECT_TRUE(df.at(a).empty());
}

TEST(GraphTest, NaturalLoops_SimpleLoop) {
    sdfg::graph::Graph graph;
    auto entry = boost::add_vertex(graph);
    auto h = boost::add_vertex(graph);
    auto body1 = boost::add_vertex(graph);
    auto exit = boost::add_vertex(graph);
    boost::add_edge(entry, h, graph);
    boost::add_edge(h, body1, graph);
    boost::add_edge(body1, h, graph);
    boost::add_edge(h, exit, graph);

    auto loops = sdfg::graph::natural_loops(graph, entry);
    ASSERT_EQ(loops.size(), 1);
    const auto& loop = loops[0];
    EXPECT_EQ(loop.header, h);
    ASSERT_EQ(loop.latches.size(), 1);
    EXPECT_EQ(loop.latches[0], body1);
    EXPECT_EQ(loop.body.size(), 2);
    EXPECT_TRUE(loop.body.find(h) != loop.body.end());
    EXPECT_TRUE(loop.body.find(body1) != loop.body.end());
    EXPECT_TRUE(loop.exits.find(exit) != loop.exits.end());
}

TEST(GraphTest, NaturalLoops_MultiLatchLoop) {
    sdfg::graph::Graph graph;
    auto entry = boost::add_vertex(graph);
    auto h = boost::add_vertex(graph);
    auto a = boost::add_vertex(graph);
    auto b = boost::add_vertex(graph);
    auto exit = boost::add_vertex(graph);
    boost::add_edge(entry, h, graph);
    boost::add_edge(h, a, graph);
    boost::add_edge(a, h, graph);
    boost::add_edge(h, b, graph);
    boost::add_edge(b, h, graph);
    boost::add_edge(h, exit, graph);

    auto loops = sdfg::graph::natural_loops(graph, entry);
    ASSERT_EQ(loops.size(), 1);
    const auto& loop = loops[0];
    EXPECT_EQ(loop.header, h);
    EXPECT_EQ(loop.latches.size(), 2);
    EXPECT_TRUE((loop.latches[0] == a && loop.latches[1] == b) || (loop.latches[0] == b && loop.latches[1] == a));
    EXPECT_EQ(loop.body.size(), 3);
    EXPECT_TRUE(loop.body.find(h) != loop.body.end());
    EXPECT_TRUE(loop.body.find(a) != loop.body.end());
    EXPECT_TRUE(loop.body.find(b) != loop.body.end());
    EXPECT_TRUE(loop.exits.find(exit) != loop.exits.end());
}

TEST(GraphTest, NaturalLoops_SelfLoop) {
    sdfg::graph::Graph graph;
    auto entry = boost::add_vertex(graph);
    auto s = boost::add_vertex(graph);
    auto exit = boost::add_vertex(graph);
    boost::add_edge(entry, s, graph);
    boost::add_edge(s, s, graph);
    boost::add_edge(s, exit, graph);

    auto loops = sdfg::graph::natural_loops(graph, entry);
    ASSERT_EQ(loops.size(), 1);
    const auto& loop = loops[0];
    EXPECT_EQ(loop.header, s);
    EXPECT_EQ(loop.latches.size(), 1);
    EXPECT_EQ(loop.latches[0], s);
    EXPECT_EQ(loop.body.size(), 1);
    EXPECT_TRUE(loop.body.find(s) != loop.body.end());
    EXPECT_TRUE(loop.exits.find(exit) != loop.exits.end());
}

TEST(GraphTest, NaturalLoops_IrreducibleMultiEntryCycle) {
    sdfg::graph::Graph g;
    auto entry = boost::add_vertex(g);
    auto a = boost::add_vertex(g);
    auto b = boost::add_vertex(g);
    auto c = boost::add_vertex(g);
    // edges
    boost::add_edge(entry, a, g);
    boost::add_edge(a, b, g);
    boost::add_edge(b, a, g); // back edge a<-b
    boost::add_edge(entry, c, g);
    boost::add_edge(c, b, g);
    boost::add_edge(c, a, g); // second entry into cycle

    auto loops = sdfg::graph::natural_loops(g, entry);
    EXPECT_EQ(loops.size(), 0) << "Irreducible multi-entry cycle should not yield natural loops";
}

TEST(GraphTest, SCCInfo_ReducibleSingleEntryLoop) {
    sdfg::graph::Graph graph;
    auto entry = boost::add_vertex(graph);
    auto a = boost::add_vertex(graph);
    auto b = boost::add_vertex(graph);
    auto c = boost::add_vertex(graph);
    auto exit = boost::add_vertex(graph);
    boost::add_edge(entry, a, graph);
    boost::add_edge(a, b, graph);
    boost::add_edge(b, c, graph);
    boost::add_edge(c, b, graph); // back edge
    boost::add_edge(c, exit, graph);
    boost::add_edge(entry, exit, graph);

    auto info = sdfg::graph::classify_sccs_irreducible(graph, entry);
    EXPECT_EQ(info.irreducible_components.size(), 0);
}

TEST(GraphTest, SCCInfo_MultiEntryCycleIsIrreducible) {
    sdfg::graph::Graph graph;
    auto entry = boost::add_vertex(graph);
    auto x = boost::add_vertex(graph);
    auto y = boost::add_vertex(graph);
    auto z = boost::add_vertex(graph);
    auto exit = boost::add_vertex(graph);
    boost::add_edge(entry, x, graph);
    boost::add_edge(x, y, graph);
    boost::add_edge(y, z, graph);
    boost::add_edge(z, x, graph); // cycle back
    boost::add_edge(entry, y, graph); // second entry into cycle
    boost::add_edge(z, exit, graph);

    auto info = sdfg::graph::classify_sccs_irreducible(graph, entry);
    // Expect one irreducible component (the cycle {x,y,z})
    EXPECT_EQ(info.irreducible_components.size(), 1);
}
