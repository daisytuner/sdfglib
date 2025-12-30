#include "sdfg/data_flow/data_flow_graph.h"

namespace sdfg {
namespace data_flow {

void DataFlowGraph::validate(const Function& function) const {
    for (auto& node : this->nodes_) {
        node.second->validate(function);
        if (&node.second->get_parent() != this) {
            throw InvalidSDFGException("DataFlowGraph: Node parent mismatch.");
        }

        if (auto code_node = dynamic_cast<const data_flow::CodeNode*>(node.second.get())) {
            if (this->in_degree(*code_node) != code_node->inputs().size()) {
                throw InvalidSDFGException("DataFlowGraph: Number of input edges does not match number of inputs.");
            }
            if (this->out_degree(*code_node) != code_node->outputs().size()) {
                throw InvalidSDFGException("DataFlowGraph: Number of output edges does not match number of outputs.");
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

std::list<const data_flow::DataFlowNode*> DataFlowGraph::topological_sort_deterministic() const {
    auto [num_components, components_map] = graph::weakly_connected_components(this->graph_);

    // Build deterministic topological sort for each weakly connected component
    std::vector<std::list<const DataFlowNode*>> components(num_components);
    for (size_t i = 0; i < num_components; i++) {
        // Get all sinks of the current component
        std::vector<const DataFlowNode*> sinks;
        bool component_empty = true;
        for (auto [v, comp] : components_map) {
            if (comp == i) {
                component_empty = false;
                if (boost::out_degree(v, this->graph_) == 0) {
                    sinks.push_back(this->nodes_.at(v).get());
                }
            }
        }
        if (sinks.size() == 0) {
            if (component_empty) {
                continue;
            } else {
                throw boost::not_a_dag();
            }
        }

        // Go over all sinks
        const DataFlowNode* primary_sink = nullptr;
        std::unordered_map<const DataFlowNode*, std::list<const DataFlowNode*>> lists;
        std::unordered_map<const DataFlowNode*, std::list<const Memlet*>> memlet_queues;
        std::unordered_map<const Memlet*, const DataFlowNode*> memlet_map;
        for (const auto* sink : sinks) {
            lists.insert({sink, {}});
            memlet_queues.insert({sink, {}});
            std::unordered_set<const DataFlowNode*> primary_blocker;

            // Perform reversed DFS starting form sink node
            std::unordered_set<const DataFlowNode*> visited;
            std::stack<std::pair<const DataFlowNode*, const DataFlowNode*>> stack({{sink, nullptr}});
            while (!stack.empty()) {
                const auto [current, successor] = stack.top();
                stack.pop();

                // Special case: Only handle primary edges
                if (this->out_degree(*current) > 1) {
                    if (const auto* code_node = dynamic_cast<const CodeNode*>(current)) {
                        std::unordered_map<std::string, const Memlet*> edges_map;
                        const Memlet* memlet = nullptr; // Memlet from current to successor
                        for (const auto& oedge : this->out_edges(*code_node)) {
                            edges_map.insert({oedge.src_conn(), &oedge});
                            if (&oedge.dst() == successor) {
                                memlet = &oedge;
                            }
                        }
                        const auto* primary_dst = &edges_map.at(code_node->output(0))->dst();
                        if (primary_dst == successor) {
                            for (size_t j = 1; j < code_node->outputs().size(); j++) {
                                const auto* edge = edges_map.at(code_node->output(j));
                                if (&edge->dst() != successor) {
                                    memlet_queues.at(sink).push_back(edge);
                                }
                            }
                        } else {
                            if (primary_blocker.empty()) {
                                memlet_map.insert({memlet, sink});
                            } else {
                                memlet_map.insert({memlet, nullptr});
                            }
                            primary_blocker.insert(current);
                            continue;
                        }
                    } else {
                        std::vector<std::pair<const Memlet*, size_t>> edges_list;
                        const Memlet* memlet = nullptr; // Memlet from current to successor
                        for (const auto& oedge : this->out_edges(*current)) {
                            const auto* dst = &oedge.dst();
                            size_t value = 0;
                            if (const auto* tasklet = dynamic_cast<const Tasklet*>(dst)) {
                                value = tasklet->code();
                            } else if (const auto* libnode = dynamic_cast<const LibraryNode*>(dst)) {
                                value = 52;
                                for (char c : libnode->code().value()) {
                                    value += c;
                                }
                            }
                            edges_list.push_back({&oedge, value});
                            if (&oedge.dst() == successor) {
                                memlet = &oedge;
                            }
                        }
                        std::sort(edges_list.begin(), edges_list.end(), [](const auto& a, const auto& b) {
                            return a.second > b.second ||
                                   (a.second == b.second && a.first->element_id() < b.first->element_id());
                        });
                        const auto* primary_dst = &edges_list.front().first->dst();
                        if (primary_dst == successor) {
                            for (size_t j = 1; j < edges_list.size(); j++) {
                                const auto* edge = edges_list.at(j).first;
                                if (&edge->dst() != successor) {
                                    memlet_queues.at(sink).push_back(edge);
                                }
                            }
                        } else {
                            if (primary_blocker.empty()) {
                                memlet_map.insert({memlet, sink});
                            } else {
                                memlet_map.insert({memlet, nullptr});
                            }
                            primary_blocker.insert(current);
                            continue;
                        }
                    }
                }

                // Put the current element in the list
                if (visited.contains(current)) {
                    continue;
                }
                visited.insert(current);
                if (primary_blocker.contains(current)) {
                    primary_blocker.erase(current);
                }
                lists.at(sink).push_front(current);

                // Put all predecessors on the stack
                if (const auto* code_node = dynamic_cast<const CodeNode*>(current)) {
                    std::unordered_set<const DataFlowNode*> pushed_predecessors;
                    for (const auto& input : code_node->inputs()) {
                        const Memlet* iedge = nullptr;
                        for (auto& in_edge : this->in_edges(*code_node)) {
                            if (in_edge.dst_conn() == input) {
                                iedge = &in_edge;
                                break;
                            }
                        }
                        if (!iedge) {
                            continue;
                        }
                        const auto* src = &iedge->src();
                        if (pushed_predecessors.contains(src)) {
                            continue;
                        }
                        stack.push({src, current});
                        pushed_predecessors.insert(src);
                    }
                } else {
                    std::vector<std::pair<const DataFlowNode*, size_t>> tmp_inputs;
                    for (const auto& iedge : this->in_edges(*current)) {
                        const auto* src = &iedge.src();
                        size_t value = 0;
                        if (const auto* tasklet = dynamic_cast<const Tasklet*>(src)) {
                            value = tasklet->code();
                        } else if (const auto* libnode = dynamic_cast<const LibraryNode*>(src)) {
                            value = 52;
                            for (char c : libnode->code().value()) {
                                value += c;
                            }
                        }
                        tmp_inputs.push_back({src, value});
                    }
                    std::sort(tmp_inputs.begin(), tmp_inputs.end(), [](const auto& a, const auto& b) {
                        return a.second > b.second ||
                               (a.second == b.second && a.first->element_id() < b.first->element_id());
                    });
                    std::unordered_set<const DataFlowNode*> pushed_predecessors;
                    for (const auto& tmp_input : tmp_inputs) {
                        if (pushed_predecessors.contains(tmp_input.first)) {
                            continue;
                        }
                        stack.push({tmp_input.first, current});
                        pushed_predecessors.insert(tmp_input.first);
                    }
                }
            }

            // Store primary sink
            if (primary_blocker.empty()) {
                primary_sink = sink;
            }
        }
        if (!primary_sink) {
            throw boost::not_a_dag();
        }

        std::list<const DataFlowNode*> queue = {primary_sink};
        std::unordered_set<const DataFlowNode*> visited;
        while (!queue.empty()) {
            const auto* current = queue.front();
            queue.pop_front();
            if (visited.contains(current)) {
                continue;
            }

            // Fill global list
            components.at(i).insert(components.at(i).end(), lists.at(current).begin(), lists.at(current).end());
            visited.insert(current);

            // Fill queue
            for (const auto* memlet : memlet_queues.at(current)) {
                const auto* node = memlet_map.at(memlet);
                if (node) {
                    queue.push_back(node);
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
