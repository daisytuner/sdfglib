#include "sdfg/data_flow/data_flow_graph.h"

namespace sdfg {
namespace data_flow {

const Element* DataFlowGraph::get_parent() const { return this->parent_; };

Element* DataFlowGraph::get_parent() { return this->parent_; };

size_t DataFlowGraph::in_degree(const data_flow::DataFlowNode& node) const {
    return boost::in_degree(node.vertex(), this->graph_);
};

size_t DataFlowGraph::out_degree(const data_flow::DataFlowNode& node) const {
    return boost::out_degree(node.vertex(), this->graph_);
};

void DataFlowGraph::replace(const symbolic::Expression& old_expression,
                            const symbolic::Expression& new_expression) {
    for (auto& node : this->nodes_) {
        node.second->replace(old_expression, new_expression);
    }

    for (auto& edge : this->edges_) {
        edge.second->replace(old_expression, new_expression);
    }
};

/***** Section: Analysis *****/

std::unordered_set<const data_flow::Tasklet*> DataFlowGraph::tasklets() const {
    std::unordered_set<const data_flow::Tasklet*> ts;
    for (auto& node : this->nodes_) {
        if (auto tasklet = dynamic_cast<const data_flow::Tasklet*>(node.second.get())) {
            ts.insert(tasklet);
        }
    }

    return ts;
};

std::unordered_set<const data_flow::AccessNode*> DataFlowGraph::data_nodes() const {
    std::unordered_set<const data_flow::AccessNode*> dnodes;
    for (auto& node : this->nodes_) {
        if (auto access_node = dynamic_cast<const data_flow::AccessNode*>(node.second.get())) {
            dnodes.insert(access_node);
        }
    }

    return dnodes;
};

std::unordered_set<data_flow::AccessNode*> DataFlowGraph::data_nodes() {
    std::unordered_set<data_flow::AccessNode*> dnodes;
    for (auto& node : this->nodes_) {
        if (auto access_node = dynamic_cast<data_flow::AccessNode*>(node.second.get())) {
            dnodes.insert(access_node);
        }
    }

    return dnodes;
};

std::unordered_set<const data_flow::AccessNode*> DataFlowGraph::reads() const {
    std::unordered_set<const data_flow::AccessNode*> rs;
    for (auto& node : this->nodes_) {
        if (auto access_node = dynamic_cast<const data_flow::AccessNode*>(node.second.get())) {
            if (this->out_degree(*access_node) > 0) {
                rs.insert(access_node);
            }
        }
    }

    return rs;
};

std::unordered_set<const data_flow::AccessNode*> DataFlowGraph::writes() const {
    std::unordered_set<const data_flow::AccessNode*> ws;
    for (auto& node : this->nodes_) {
        if (auto access_node = dynamic_cast<const data_flow::AccessNode*>(node.second.get())) {
            if (this->in_degree(*access_node) > 0) {
                ws.insert(access_node);
            }
        }
    }

    return ws;
};

std::unordered_set<const data_flow::DataFlowNode*> DataFlowGraph::sources() const {
    std::unordered_set<const data_flow::DataFlowNode*> ss;
    for (auto& node : this->nodes_) {
        if (this->in_degree(*node.second) == 0) {
            ss.insert(node.second.get());
        }
    }

    return ss;
};

std::unordered_set<data_flow::DataFlowNode*> DataFlowGraph::sources() {
    std::unordered_set<data_flow::DataFlowNode*> ss;
    for (auto& node : this->nodes_) {
        if (this->in_degree(*node.second) == 0) {
            ss.insert(node.second.get());
        }
    }

    return ss;
};

std::unordered_set<const data_flow::DataFlowNode*> DataFlowGraph::sinks() const {
    std::unordered_set<const data_flow::DataFlowNode*> ss;
    for (auto& node : this->nodes_) {
        if (this->out_degree(*node.second) == 0) {
            ss.insert(node.second.get());
        }
    }

    return ss;
};

std::unordered_set<data_flow::DataFlowNode*> DataFlowGraph::sinks() {
    std::unordered_set<data_flow::DataFlowNode*> ss;
    for (auto& node : this->nodes_) {
        if (this->out_degree(*node.second) == 0) {
            ss.insert(node.second.get());
        }
    }

    return ss;
};

std::unordered_set<const data_flow::DataFlowNode*> DataFlowGraph::predecessors(
    const data_flow::DataFlowNode& node) const {
    std::unordered_set<const data_flow::DataFlowNode*> ss;
    for (auto& edge : this->in_edges(node)) {
        ss.insert(&edge.src());
    }

    return ss;
};

std::unordered_set<const data_flow::DataFlowNode*> DataFlowGraph::successors(
    const data_flow::DataFlowNode& node) const {
    std::unordered_set<const data_flow::DataFlowNode*> ss;
    for (auto& edge : this->out_edges(node)) {
        ss.insert(&edge.dst());
    }

    return ss;
};

std::list<const data_flow::DataFlowNode*> DataFlowGraph::topological_sort() const {
    std::list<graph::Vertex> order = graph::topological_sort(this->graph_);

    std::list<const data_flow::DataFlowNode*> topo_nodes;
    for (auto& vertex : order) {
        topo_nodes.push_back(this->nodes_.at(vertex).get());
    }

    return topo_nodes;
};

std::list<data_flow::DataFlowNode*> DataFlowGraph::topological_sort() {
    std::list<graph::Vertex> order = graph::topological_sort(this->graph_);

    std::list<data_flow::DataFlowNode*> topo_nodes;
    for (auto& vertex : order) {
        topo_nodes.push_back(this->nodes_.at(vertex).get());
    }

    return topo_nodes;
};

std::unordered_map<std::string, const data_flow::AccessNode*> DataFlowGraph::dominators() const {
    std::unordered_map<std::string, const data_flow::AccessNode*> frontier;
    for (auto& node : this->topological_sort()) {
        if (auto access_node = dynamic_cast<const data_flow::AccessNode*>(node)) {
            if (frontier.find(access_node->data()) == frontier.end()) {
                frontier[access_node->data()] = access_node;
            }
        }
    }

    return frontier;
};

std::unordered_map<std::string, const data_flow::AccessNode*> DataFlowGraph::post_dominators()
    const {
    std::unordered_map<std::string, const data_flow::AccessNode*> frontier;
    for (auto& node : this->topological_sort()) {
        if (auto access_node = dynamic_cast<const data_flow::AccessNode*>(node)) {
            frontier[access_node->data()] = access_node;
        }
    }

    return frontier;
};

std::unordered_map<std::string, data_flow::AccessNode*> DataFlowGraph::post_dominators() {
    std::unordered_map<std::string, data_flow::AccessNode*> frontier;
    for (auto& node : this->topological_sort()) {
        if (auto access_node = dynamic_cast<data_flow::AccessNode*>(node)) {
            frontier[access_node->data()] = access_node;
        }
    }

    return frontier;
};

auto DataFlowGraph::all_simple_paths(const data_flow::DataFlowNode& src,
                                     const data_flow::DataFlowNode& dst) const {
    std::list<std::list<graph::Edge>> all_paths_raw =
        graph::all_simple_paths(this->graph_, src.vertex(), dst.vertex());

    std::list<std::list<std::reference_wrapper<data_flow::Memlet>>> all_paths;
    for (auto& path_raw : all_paths_raw) {
        std::list<std::reference_wrapper<data_flow::Memlet>> path;
        for (auto& edge : path_raw) {
            path.push_back(*this->edges_.at(edge));
        }
        all_paths.push_back(path);
    }

    return all_paths;
};

const std::pair<size_t, const std::unordered_map<const data_flow::DataFlowNode*, size_t>>
DataFlowGraph::weakly_connected_components() const {
    auto ccs_vertex = graph::weakly_connected_components(this->graph_);

    std::unordered_map<const data_flow::DataFlowNode*, size_t> ccs;
    for (auto& entry : ccs_vertex.second) {
        ccs[this->nodes_.at(entry.first).get()] = entry.second;
    }

    return {ccs_vertex.first, ccs};
};

/***** Section: Serialization *****/

std::unique_ptr<DataFlowGraph> DataFlowGraph::clone() const {
    auto new_graph = std::make_unique<DataFlowGraph>();

    std::unordered_map<graph::Vertex, graph::Vertex> node_mapping;
    for (auto& entry : this->nodes_) {
        auto vertex = boost::add_vertex(new_graph->graph_);
        auto res = new_graph->nodes_.insert({vertex, entry.second->clone(vertex, *new_graph)});
        node_mapping.insert({entry.first, vertex});
    }

    for (auto& entry : this->edges_) {
        auto src = node_mapping[entry.second->src().vertex()];
        auto dst = node_mapping[entry.second->dst().vertex()];

        auto edge = boost::add_edge(src, dst, new_graph->graph_);

        auto res = new_graph->edges_.insert(
            {edge.first, entry.second->clone(edge.first, *new_graph, *new_graph->nodes_[src],
                                             *new_graph->nodes_[dst])});
    }

    return std::move(new_graph);
};

}  // namespace data_flow
}  // namespace sdfg
