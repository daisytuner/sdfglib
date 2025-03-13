#include "sdfg/sdfg.h"

#include "sdfg/codegen/language_extensions/c_language_extension.h"
#include "sdfg/element.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {

SDFG::SDFG(const std::string& name)
    : Function(name),
      start_state_(nullptr){

      };

const DebugInfo SDFG::debug_info() const {
    DebugInfo info;
    for (auto& state : this->states()) {
        info = DebugInfo::merge(info, state.debug_info());
        for (auto& data_flow_node : state.dataflow().nodes()) {
            info = DebugInfo::merge(info, data_flow_node.debug_info());
        }
        for (auto& edge : state.dataflow().edges()) {
            info = DebugInfo::merge(info, edge.debug_info());
        }
    }
    for (auto& edges : this->edges()) {
        info = DebugInfo::merge(info, edges.debug_info());
    }
    return info;
};

size_t SDFG::in_degree(const control_flow::State& state) const {
    return boost::in_degree(state.vertex(), this->graph_);
};

size_t SDFG::out_degree(const control_flow::State& state) const {
    return boost::out_degree(state.vertex(), this->graph_);
};

bool SDFG::is_adjacent(const control_flow::State& src, const control_flow::State& dst) const {
    return boost::edge(src.vertex(), dst.vertex(), this->graph_).second;
};

const control_flow::InterstateEdge& SDFG::edge(const control_flow::State& src,
                                               const control_flow::State& dst) const {
    auto e = boost::edge(src.vertex(), dst.vertex(), this->graph_);
    assert(e.second);
    return *this->edges_.at(e.first);
};

const control_flow::State& SDFG::start_state() const {
    assert(this->start_state_ != nullptr && "Start state not set");

    return *this->start_state_;
};

std::unordered_map<const control_flow::State*, const control_flow::State*> SDFG::dominator_tree()
    const {
    auto dom_tree_ = graph::dominator_tree(this->graph_, this->start_state_->vertex());

    std::unordered_map<const control_flow::State*, const control_flow::State*> dom_tree;
    for (auto& entry : dom_tree_) {
        control_flow::State* first = this->states_.at(entry.first).get();
        control_flow::State* second = nullptr;
        if (entry.second != boost::graph_traits<graph::Graph>::null_vertex()) {
            second = this->states_.at(entry.second).get();
        }
        dom_tree.insert({first, second});
    }

    return dom_tree;
};

std::unordered_map<const control_flow::State*, const control_flow::State*>
SDFG::post_dominator_tree() const {
    auto terminal_state = this->terminal_states();
    assert(std::distance(terminal_state.begin(), terminal_state.end()) == 1);

    auto pdom_tree_ = graph::post_dominator_tree(this->graph_, (*terminal_state.begin()).vertex());

    std::unordered_map<const control_flow::State*, const control_flow::State*> pdom_tree;
    for (auto& entry : pdom_tree_) {
        control_flow::State* first = this->states_.at(entry.first).get();
        control_flow::State* second = nullptr;
        if (entry.second != boost::graph_traits<graph::Graph>::null_vertex()) {
            second = this->states_.at(entry.second).get();
        }
        pdom_tree.insert({first, second});
    }

    return pdom_tree;
};

std::list<const control_flow::InterstateEdge*> SDFG::back_edges() const {
    std::list<const control_flow::InterstateEdge*> bedges;
    for (const auto& edge : graph::back_edges(this->graph_, this->start_state_->vertex())) {
        bedges.push_back(this->edges_.find(edge)->second.get());
    }

    return bedges;
};

std::list<std::list<const control_flow::InterstateEdge*>> SDFG::all_simple_paths(
    const control_flow::State& src, const control_flow::State& dst) const {
    std::list<std::list<const control_flow::InterstateEdge*>> all_paths;

    std::list<std::list<graph::Edge>> all_paths_raw =
        graph::all_simple_paths(this->graph_, src.vertex(), dst.vertex());
    for (auto& path_raw : all_paths_raw) {
        std::list<const control_flow::InterstateEdge*> path;
        for (auto& edge : path_raw) {
            path.push_back(this->edges_.find(edge)->second.get());
        }
        all_paths.push_back(path);
    }

    return all_paths;
};

void SDFG::as_dot(std::ostream& f) const {
    std::map<graph::Graph::vertex_descriptor, size_t> node_ids;
    std::map<graph::Graph::vertex_descriptor, std::string> node_names;
    for (auto u : boost::make_iterator_range(boost::vertices(this->graph_))) {
        node_ids[u] = node_ids.size();
        node_names[u] = this->states_.at(u)->name();
    }

    codegen::CLanguageExtension lang_ext;
    std::map<graph::Graph::edge_descriptor, std::string> edge_names;
    for (auto u : boost::make_iterator_range(boost::edges(this->graph_))) {
        std::string desc = lang_ext.expression(this->edges_.at(u)->condition());
        desc += " ; ";
        for (auto& ass : this->edges_.at(u)->assignments()) {
            desc += lang_ext.expression(ass.first) + " = " + lang_ext.expression(ass.second) + ",";
        }
        edge_names[u] = desc;
    }

    boost::default_writer w;
    boost::write_graphviz(f, this->graph_,
                          boost::make_label_writer(boost::make_assoc_property_map(node_names)),
                          boost::make_label_writer(boost::make_assoc_property_map(edge_names)), w,
                          boost::make_assoc_property_map(node_ids));
};

}  // namespace sdfg
