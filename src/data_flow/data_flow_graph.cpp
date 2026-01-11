#include "sdfg/data_flow/data_flow_graph.h"

#include <algorithm>
#include <queue>

namespace sdfg {
namespace data_flow {

void DataFlowGraph::validate(const Function& function) const {
    for (auto& node : this->nodes_) {
        node.second->validate(function);
        if (&node.second->get_parent() != this) {
            throw InvalidSDFGException("DataFlowGraph: Node parent mismatch.");
        }

        // No two access nodes for same data
        std::unordered_map<std::string, const AccessNode*> input_names;
        for (auto& iedge : this->in_edges(*node.second)) {
            if (dynamic_cast<const ConstantNode*>(&iedge.src()) != nullptr) {
                continue;
            }
            auto& src = static_cast<const AccessNode&>(iedge.src());
            if (input_names.find(src.data()) != input_names.end()) {
                if (input_names.at(src.data()) != &src) {
                    throw InvalidSDFGException("Two access nodes with the same data as iedge: " + src.data());
                }
            } else {
                input_names.insert({src.data(), &src});
            }
        }
    }
    for (auto& edge : this->edges_) {
        edge.second->validate(function);
    }
};

const Element* DataFlowGraph::get_parent() const { return this->parent_; };

Element* DataFlowGraph::get_parent() { return this->parent_; };

size_t DataFlowGraph::in_degree(const data_flow::DataFlowNode& node) const {
    return boost::in_degree(node.vertex(), this->graph_);
};

size_t DataFlowGraph::out_degree(const data_flow::DataFlowNode& node) const {
    return boost::out_degree(node.vertex(), this->graph_);
};

void DataFlowGraph::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
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

std::unordered_set<data_flow::Tasklet*> DataFlowGraph::tasklets() {
    std::unordered_set<data_flow::Tasklet*> ts;
    for (auto& node : this->nodes_) {
        if (auto tasklet = dynamic_cast<data_flow::Tasklet*>(node.second.get())) {
            ts.insert(tasklet);
        }
    }

    return ts;
};

std::unordered_set<const data_flow::LibraryNode*> DataFlowGraph::library_nodes() const {
    std::unordered_set<const data_flow::LibraryNode*> ls;
    for (auto& node : this->nodes_) {
        if (auto lib_node = dynamic_cast<const data_flow::LibraryNode*>(node.second.get())) {
            ls.insert(lib_node);
        }
    }

    return ls;
};

std::unordered_set<data_flow::LibraryNode*> DataFlowGraph::library_nodes() {
    std::unordered_set<data_flow::LibraryNode*> ls;
    for (auto& node : this->nodes_) {
        if (auto lib_node = dynamic_cast<data_flow::LibraryNode*>(node.second.get())) {
            ls.insert(lib_node);
        }
    }

    return ls;
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

std::unordered_set<const data_flow::DataFlowNode*> DataFlowGraph::predecessors(const data_flow::DataFlowNode& node
) const {
    std::unordered_set<const data_flow::DataFlowNode*> ss;
    for (auto& edge : this->in_edges(node)) {
        ss.insert(&edge.src());
    }

    return ss;
};

std::unordered_set<const data_flow::DataFlowNode*> DataFlowGraph::successors(const data_flow::DataFlowNode& node
) const {
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

std::unordered_map<std::string, const data_flow::AccessNode*> DataFlowGraph::post_dominators() const {
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

auto DataFlowGraph::all_simple_paths(const data_flow::DataFlowNode& src, const data_flow::DataFlowNode& dst) const {
    std::list<std::list<graph::Edge>> all_paths_raw = graph::all_simple_paths(this->graph_, src.vertex(), dst.vertex());

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

const std::pair<size_t, const std::unordered_map<const data_flow::DataFlowNode*, size_t>> DataFlowGraph::
    weakly_connected_components() const {
    auto ccs_vertex = graph::weakly_connected_components(this->graph_);

    std::unordered_map<const data_flow::DataFlowNode*, size_t> ccs;
    for (auto& entry : ccs_vertex.second) {
        ccs[this->nodes_.at(entry.first).get()] = entry.second;
    }

    return {ccs_vertex.first, ccs};
};

// Helper function to get primary outgoing edge for a node with multiple outputs
const data_flow::Memlet* get_primary_outgoing_edge(const DataFlowGraph& graph, const data_flow::DataFlowNode* node) {
    if (const auto* code_node = dynamic_cast<const data_flow::CodeNode*>(node)) {
        // For CodeNodes: first output is primary
        if (code_node->outputs().empty()) {
            return nullptr;
        }

        for (const auto& oedge : graph.out_edges(*code_node)) {
            if (oedge.src_conn() == code_node->output(0)) {
                return &oedge;
            }
        }
        return nullptr;
    } else {
        // For other nodes: highest priority edge (by tasklet code or lib name)
        std::vector<std::pair<const data_flow::Memlet*, size_t>> edges_list;
        for (const auto& oedge : graph.out_edges(*node)) {
            const auto* dst = &oedge.dst();
            size_t value = 0;
            if (const auto* tasklet = dynamic_cast<const data_flow::Tasklet*>(dst)) {
                value = tasklet->code();
            } else if (const auto* libnode = dynamic_cast<const data_flow::LibraryNode*>(dst)) {
                value = 52;
                for (char c : libnode->code().value()) {
                    value += c;
                }
            }
            edges_list.push_back({&oedge, value});
        }

        if (!edges_list.empty()) {
            std::sort(edges_list.begin(), edges_list.end(), [](const auto& a, const auto& b) {
                return a.second > b.second || (a.second == b.second && a.first->element_id() < b.first->element_id());
            });
            return edges_list.front().first;
        }
    }
    return nullptr;
}

std::list<const data_flow::DataFlowNode*> DataFlowGraph::topological_sort_deterministic() const {
    auto [num_components, components_map] = graph::weakly_connected_components(this->graph_);

    // Build deterministic topological sort for each weakly connected component
    std::vector<std::list<const DataFlowNode*>> components(num_components);

    for (size_t i = 0; i < num_components; i++) {
        // Get all nodes in this component
        std::vector<const DataFlowNode*> component_nodes;
        for (auto [v, comp] : components_map) {
            if (comp == i) {
                component_nodes.push_back(this->nodes_.at(v).get());
            }
        }

        if (component_nodes.empty()) {
            continue;
        }

        // Check for cycles: if no sinks exist, it's a cycle
        bool has_sink = false;
        for (const auto* node : component_nodes) {
            if (boost::out_degree(node->vertex(), this->graph_) == 0) {
                has_sink = true;
                break;
            }
        }
        if (!has_sink) {
            throw boost::not_a_dag();
        }

        // New algorithm: Hybrid Kahn's with priority-based processing

        // Step 1: Initialize in-degree and primary_incoming_count
        std::unordered_map<const DataFlowNode*, size_t> in_degree;
        std::unordered_map<const DataFlowNode*, size_t> primary_incoming_count;

        for (const auto* node : component_nodes) {
            size_t count = 0;
            for (auto& edge : this->in_edges(*node)) {
                (void) edge; // Just count
                count++;
            }
            in_degree[node] = count;
            primary_incoming_count[node] = 0;
        }

        // Step 2: Mark primary edges
        for (const auto* node : component_nodes) {
            if (this->out_degree(*node) > 1) {
                const Memlet* primary_edge = get_primary_outgoing_edge(*this, node);
                if (primary_edge) {
                    primary_incoming_count[&primary_edge->dst()]++;
                }
            } else if (this->out_degree(*node) == 1) {
                auto edges = this->out_edges(*node);
                auto it = edges.begin();
                if (it != edges.end()) {
                    primary_incoming_count[&(*it).dst()]++;
                }
            }
        }

        // Step 3: Priority queue for node ordering
        // Use a struct to define priority
        struct NodePriority {
            const DataFlowNode* node;
            size_t primary_path_count;
            size_t element_id;

            bool operator<(const NodePriority& other) const {
                // Higher primary_path_count = higher priority (use > for max-heap behavior in std::priority_queue)
                if (primary_path_count != other.primary_path_count)
                    return primary_path_count < other.primary_path_count;
                // Lower element_id = higher priority
                return element_id > other.element_id;
            }
        };

        std::priority_queue<NodePriority> queue;

        // Add all nodes with in-degree 0 to the queue
        for (const auto* node : component_nodes) {
            if (in_degree[node] == 0) {
                queue.push({node, primary_incoming_count[node], node->element_id()});
            }
        }

        // Step 4: Process nodes
        while (!queue.empty()) {
            NodePriority current = queue.top();
            queue.pop();

            components.at(i).push_back(current.node);

            // Update successors
            for (auto& edge : this->out_edges(*current.node)) {
                const DataFlowNode* successor = &edge.dst();
                in_degree[successor]--;

                if (in_degree[successor] == 0) {
                    queue.push({successor, primary_incoming_count[successor], successor->element_id()});
                }
            }
        }
    }

    // Sort components
    std::sort(components.begin(), components.end(), [](const auto& a, const auto& b) {
        return a.size() > b.size() ||
               (a.size() == b.size() && a.size() > 0 && a.front()->element_id() < b.front()->element_id());
    });

    // Resulting data structure
    std::list<const DataFlowNode*> order;
    for (auto& component : components) {
        order.insert(order.end(), component.begin(), component.end());
    }

    return order;
};


} // namespace data_flow
} // namespace sdfg
