#include "sdfg/graph/graph.h"

namespace sdfg {
namespace graph {

const ReverseGraph reverse(const Graph& graph) { return boost::reverse_graph<Graph>(graph); };

const std::tuple<
    std::unique_ptr<const UndirectedGraph>,
    const std::unordered_map<Vertex, Vertex>,
    const std::unordered_map<Vertex, Vertex>>
undirected(const Graph& graph) {
    auto undirected_graph = std::make_unique<UndirectedGraph>();
    std::unordered_map<Vertex, Vertex> mapping;
    std::unordered_map<Vertex, Vertex> reverse_mapping;

    boost::graph_traits<Graph>::vertex_iterator vi, vend;
    for (boost::tie(vi, vend) = boost::vertices(graph); vi != vend; ++vi) {
        auto new_vertex = boost::add_vertex(*undirected_graph);
        mapping.emplace(*vi, new_vertex);
        reverse_mapping.emplace(new_vertex, *vi);
    }

    auto [eb, ee] = boost::edges(graph);
    auto edges = std::ranges::subrange(eb, ee);
    for (const auto& edge : edges) {
        const auto& u = boost::source(edge, graph);
        const auto& v = boost::target(edge, graph);
        boost::add_edge(mapping.at(u), mapping.at(v), *undirected_graph);
    }

    return {std::move(undirected_graph), mapping, reverse_mapping};
};

const std::list<Vertex> depth_first_search(const Graph& graph, const graph::Vertex start) {
    std::list<Vertex> nodes;
    std::list<Edge> back_edges;
    DFSVisitor visitor(back_edges, nodes);

    std::unordered_map<graph::Vertex, boost::default_color_type> vertex_colors;
    boost::depth_first_search(graph, visitor, boost::make_assoc_property_map(vertex_colors), start);

    return nodes;
};

std::pair<int, std::unordered_map<Vertex, size_t>> strongly_connected_components(const Graph& graph) {
    IndexMap vertex_index_map;
    boost::associative_property_map<IndexMap> boost_index_map(vertex_index_map);
    boost::graph_traits<Graph>::vertex_iterator vi, vend;
    size_t i = 0;
    for (boost::tie(vi, vend) = boost::vertices(graph); vi != vend; ++vi, ++i) {
        boost::put(boost_index_map, *vi, i);
    }

    std::unordered_map<Vertex, size_t> component_map;
    boost::associative_property_map<IndexMap> boost_component_map(component_map);

    size_t num_ccs = boost::strong_components(graph, boost_component_map, boost::vertex_index_map(boost_index_map));

    return {num_ccs, component_map};
};

std::pair<size_t, const std::unordered_map<Vertex, size_t>> weakly_connected_components(const Graph& graph) {
    const auto& undirected_graph = undirected(graph);

    IndexMap vertex_index_map;
    boost::associative_property_map<IndexMap> boost_index_map(vertex_index_map);
    boost::graph_traits<UndirectedGraph>::vertex_iterator vi, vend;
    size_t i = 0;
    for (boost::tie(vi, vend) = boost::vertices(*std::get<0>(undirected_graph)); vi != vend; ++vi, ++i) {
        boost::put(boost_index_map, *vi, i);
    }

    std::unordered_map<Vertex, size_t> undirected_component_map;
    boost::associative_property_map<std::unordered_map<Vertex, size_t>> boost_component_map(undirected_component_map);

    size_t num_ccs = boost::connected_components(
        *std::get<0>(undirected_graph), boost_component_map, boost::vertex_index_map(boost_index_map)
    );

    std::unordered_map<Vertex, size_t> component_map;
    for (const auto& entry : undirected_component_map) {
        component_map.emplace(std::get<2>(undirected_graph).at(entry.first), entry.second);
    }

    return {num_ccs, component_map};
};

const std::unordered_map<Vertex, Vertex> dominator_tree(const Graph& graph, const Vertex src) {
    std::unordered_map<Vertex, Vertex> dom_tree;

    IndexMap vertex_index_map;
    boost::associative_property_map<IndexMap> boost_index_map(vertex_index_map);
    boost::graph_traits<Graph>::vertex_iterator vi, vend;
    size_t i = 0;
    for (boost::tie(vi, vend) = boost::vertices(graph); vi != vend; ++vi, ++i) {
        boost::put(boost_index_map, *vi, i);
    }

    std::vector<size_t> df_num(num_vertices(graph), 0);
    auto df_num_map(boost::make_iterator_property_map(df_num.begin(), boost_index_map));

    std::vector<Vertex> parent(num_vertices(graph), boost::graph_traits<Graph>::null_vertex());
    auto parent_map(boost::make_iterator_property_map(parent.begin(), boost_index_map));

    std::vector<Vertex> dom_tree_vec(num_vertices(graph), boost::graph_traits<Graph>::null_vertex());
    auto dom_tree_pred_map(boost::make_iterator_property_map(dom_tree_vec.begin(), boost_index_map));

    std::vector<Vertex> vertices_by_df_num(parent);

    boost::lengauer_tarjan_dominator_tree(
        graph, src, boost_index_map, df_num_map, parent_map, vertices_by_df_num, dom_tree_pred_map
    );

    for (const auto& entry : vertex_index_map) {
        dom_tree.insert({entry.first, dom_tree_vec[entry.second]});
    }

    return dom_tree;
};

const std::unordered_map<Vertex, Vertex> post_dominator_tree(Graph& graph) {
    // Determine terminal vertices
    std::unordered_set<Vertex> terminal_vertices;
    for (auto [vb, ve] = boost::vertices(graph); vb != ve; ++vb) {
        if (boost::out_degree(*vb, graph) == 0) {
            terminal_vertices.insert(*vb);
        }
    }
    assert(!terminal_vertices.empty());

    // add synthetic super-terminal if needed
    bool modified = false;
    graph::Vertex src;
    if (terminal_vertices.size() == 1) {
        src = *terminal_vertices.begin();
        modified = false;
    } else {
        src = boost::add_vertex(graph);
        for (const auto& v : terminal_vertices) {
            boost::add_edge(v, src, graph);
        }
        modified = true;
    }


    auto& rgraph = reverse(graph);

    std::unordered_map<Vertex, Vertex> pdom_tree;

    IndexMap vertex_index_map;
    boost::associative_property_map<IndexMap> boost_index_map(vertex_index_map);
    boost::graph_traits<Graph>::vertex_iterator vi, vend;
    size_t i = 0;
    for (boost::tie(vi, vend) = boost::vertices(rgraph); vi != vend; ++vi, ++i) {
        boost::put(boost_index_map, *vi, i);
    }

    std::vector<size_t> df_num(num_vertices(rgraph), 0);
    auto df_num_map(boost::make_iterator_property_map(df_num.begin(), boost_index_map));

    std::vector<Vertex> parent(num_vertices(rgraph), boost::graph_traits<Graph>::null_vertex());
    auto parent_map(boost::make_iterator_property_map(parent.begin(), boost_index_map));

    std::vector<Vertex> pdom_tree_vec(num_vertices(rgraph), boost::graph_traits<Graph>::null_vertex());
    auto pdom_tree_pred_map(boost::make_iterator_property_map(pdom_tree_vec.begin(), boost_index_map));

    std::vector<Vertex> vertices_by_df_num(parent);

    boost::lengauer_tarjan_dominator_tree(
        rgraph, src, boost_index_map, df_num_map, parent_map, vertices_by_df_num, pdom_tree_pred_map
    );

    for (const auto& entry : vertex_index_map) {
        pdom_tree.insert({entry.first, pdom_tree_vec[entry.second]});
    }

    // Remove synthetic super-terminal if added
    if (modified) {
        boost::clear_vertex(src, graph);
    }

    return pdom_tree;
};

const std::unordered_map<Vertex, std::unordered_set<Vertex>>
dominance_frontiers(const Graph& graph, const std::unordered_map<Vertex, Vertex>& idom) {
    // Build children list from idom relation
    std::unordered_map<Vertex, std::vector<Vertex>> dom_children;
    for (const auto& [v, i] : idom) {
        if (i != boost::graph_traits<Graph>::null_vertex()) {
            dom_children[i].push_back(v);
        }
    }

    // Post-order traversal of dominator tree
    std::vector<Vertex> post_order;
    std::function<void(Vertex)> dfs_dom = [&](Vertex v) {
        for (auto c : dom_children[v]) {
            dfs_dom(c);
        }
        post_order.push_back(v);
    };
    // Find root (the one that is not dominated by anyone)
    Vertex root = boost::graph_traits<Graph>::null_vertex();
    for (const auto& [v, i] : idom) {
        if (i == boost::graph_traits<Graph>::null_vertex()) {
            root = v;
            break;
        }
    }
    if (root != boost::graph_traits<Graph>::null_vertex()) {
        dfs_dom(root);
    }

    std::unordered_map<Vertex, std::unordered_set<Vertex>> frontiers;
    for (const auto& [v, _] : idom) {
        frontiers.emplace(v, std::unordered_set<Vertex>());
    }

    auto dominates = [&](Vertex a, Vertex b) -> bool {
        if (a == b) return true;
        auto it = idom.find(b);
        while (it != idom.end() && it->second != boost::graph_traits<Graph>::null_vertex()) {
            if (it->second == a) return true;
            auto next = idom.find(it->second);
            if (next == idom.end()) break;
            it = next;
        }
        return false;
    };

    // Local frontiers: successors whose idom is not the node
    for (const auto& [b, _] : idom) {
        auto [eb, ee] = boost::out_edges(b, graph);
        auto edges = std::ranges::subrange(eb, ee);
        for (auto e : edges) {
            Vertex succ = boost::target(e, graph);
            if (idom.find(succ) == idom.end()) continue; // skip synthetic
            if (idom.at(succ) != b) {
                frontiers[b].insert(succ);
            }
        }
    }

    // Up frontiers: propagate from children with dominance check
    for (auto b : post_order) {
        for (auto c : dom_children[b]) {
            for (auto w : frontiers[c]) {
                if (!dominates(b, w) || b == w) {
                    frontiers[b].insert(w);
                }
            }
        }
    }

    return frontiers;
}

const std::list<graph::Vertex> topological_sort(const Graph& graph) {
    std::unordered_map<graph::Vertex, boost::default_color_type> vertex_colors;
    std::list<graph::Vertex> order;
    boost::topological_sort(
        graph, std::back_inserter(order), boost::color_map(boost::make_assoc_property_map(vertex_colors))
    );
    order.reverse();
    return order;
};

bool is_acyclic(const Graph& graph) {
    try {
        topological_sort(graph);
        return true;
    } catch (boost::not_a_dag e) {
        return false;
    }
};

const std::list<Edge> back_edges(const Graph& graph, const graph::Vertex start) {
    std::list<Vertex> nodes;
    std::list<Edge> back_edges;
    DFSVisitor visitor(back_edges, nodes);

    std::unordered_map<graph::Vertex, boost::default_color_type> vertex_colors;
    boost::depth_first_search(graph, visitor, boost::make_assoc_property_map(vertex_colors), start);

    return back_edges;
};

SCCInfo classify_sccs_irreducible(const Graph& graph, Vertex entry) {
    // Strongly connected components
    auto [num, comp_map] = strongly_connected_components(graph);
    SCCInfo info;
    info.num_components = num;
    info.component_of = comp_map;

    // Collect vertices per component
    for (auto [v_it, v_end] = boost::vertices(graph); v_it != v_end; ++v_it) {
        auto v = *v_it;
        auto cid = comp_map.at(v);
        info.component_vertices[cid].insert(v);
    }

    // Count entry edges per component (edges from outside component to inside)
    std::unordered_map<size_t, size_t> entry_edge_count;
    for (const auto& [cid, verts] : info.component_vertices) {
        entry_edge_count[cid] = 0;
    }
    for (auto [v_it, v_end] = boost::vertices(graph); v_it != v_end; ++v_it) {
        auto v = *v_it;
        auto [eb, ee] = boost::out_edges(v, graph);
        for (auto e_it = eb; e_it != ee; ++e_it) {
            auto tgt = boost::target(*e_it, graph);
            auto c_src = comp_map.at(v);
            auto c_tgt = comp_map.at(tgt);
            if (c_src != c_tgt) {
                // Edge crosses components -> counts toward target component entry edges
                entry_edge_count[c_tgt] += 1;
            }
        }
    }

    // Mark irreducible: component is cyclic (size>1 or self-loop) and has >1 distinct entry edges
    for (const auto& [cid, verts] : info.component_vertices) {
        bool cyclic = false;
        if (verts.size() > 1) {
            cyclic = true;
        } else {
            // single vertex with self-loop
            auto v = *verts.begin();
            auto [eb, ee] = boost::out_edges(v, graph);
            for (auto e_it = eb; e_it != ee; ++e_it) {
                if (boost::target(*e_it, graph) == v) {
                    cyclic = true;
                    break;
                }
            }
        }
        if (!cyclic) continue;
        if (entry_edge_count[cid] > 1) {
            info.irreducible_components.insert(cid);
        }
    }

    return info;
}

std::vector<NaturalLoop> natural_loops(const Graph& graph, Vertex entry) {
    std::vector<NaturalLoop> loops;
    auto idom = dominator_tree(graph, entry);
    auto backs = back_edges(graph, entry);
    auto scc_info = classify_sccs_irreducible(graph, entry);

    // Group back edges by header
    std::unordered_map<Vertex, std::vector<Vertex>> header_to_latches;
    auto dominates = [&](Vertex a, Vertex b) -> bool { // a dominates b if walk via idom chain reaches a
        if (a == b) return true;
        auto it = idom.find(b);
        while (it != idom.end() && it->second != boost::graph_traits<Graph>::null_vertex()) {
            if (it->second == a) return true;
            it = idom.find(it->second);
        }
        return false;
    };
    for (auto e : backs) {
        Vertex tail = boost::source(e, graph);
        Vertex head = boost::target(e, graph);
        bool accept = false;
        if (dominates(head, tail)) {
            accept = true;
        } else {
            auto cid_head = scc_info.component_of.at(head);
            auto cid_tail = scc_info.component_of.at(tail);
            if (cid_head == cid_tail &&
                scc_info.irreducible_components.find(cid_head) == scc_info.irreducible_components.end()) {
                accept = true; // reducible cycle but dominance check failed; accept conservatively
            }
        }
        if (!accept) continue;
        header_to_latches[head].push_back(tail);
    }

    // Build loop bodies
    for (auto& [header, latches] : header_to_latches) {
        std::unordered_set<Vertex> body;
        body.insert(header);
        for (auto latch : latches) body.insert(latch);

        // Reverse DFS from each latch up to header collecting predecessors
        std::function<void(Vertex)> collect = [&](Vertex v) {
            if (body.find(v) != body.end()) return;
            body.insert(v);
            auto [ib, ie] = boost::in_edges(v, graph);
            for (auto e_it = ib; e_it != ie; ++e_it) {
                auto pred = boost::source(*e_it, graph);
                // Only traverse if header dominates pred (stay within natural loop region)
                Vertex cur = pred;
                bool dominated = false;
                while (cur != boost::graph_traits<Graph>::null_vertex()) {
                    if (cur == header) {
                        dominated = true;
                        break;
                    }
                    auto it = idom.find(cur);
                    if (it == idom.end()) break;
                    cur = it->second;
                }
                if (dominated) collect(pred);
            }
        };
        for (auto latch : latches) collect(latch);

        // Determine exits
        std::unordered_set<Vertex> exits;
        for (auto v : body) {
            auto [eb, ee] = boost::out_edges(v, graph);
            for (auto e_it = eb; e_it != ee; ++e_it) {
                auto succ = boost::target(*e_it, graph);
                if (body.find(succ) == body.end()) exits.insert(succ);
            }
        }

        loops.push_back(NaturalLoop{header, latches, std::move(body), std::move(exits)});
    }

    return loops;
}

} // namespace graph
} // namespace sdfg
