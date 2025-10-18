#include "sdfg/analysis/dominance_analysis.h"

namespace sdfg {
namespace analysis {

DominanceAnalysis::DominanceAnalysis(StructuredSDFG& sdfg) : Analysis(sdfg) {}

void DominanceAnalysis::run(AnalysisManager& analysis_manager) {
    this->dom_tree_.clear();
    this->pdom_tree_.clear();

    auto& users_analysis = analysis_manager.get<analysis::Users>();

    // Add artifical super source and super sink
    auto super_source = boost::add_vertex(users_analysis.graph_);
    for (auto& entry : users_analysis.users_) {
        if (boost::in_degree(entry.first, users_analysis.graph_) == 0) {
            boost::add_edge(super_source, entry.first, users_analysis.graph_);
        }
    }
    auto super_sink = boost::add_vertex(users_analysis.graph_);
    for (auto& entry : users_analysis.users_) {
        if (boost::out_degree(entry.first, users_analysis.graph_) == 0) {
            boost::add_edge(entry.first, super_sink, users_analysis.graph_);
        }
    }

    this->dom_tree_ = graph::dominator_tree(users_analysis.graph_, super_source);
    this->pdom_tree_ = graph::post_dominator_tree(users_analysis.graph_);

    for (auto& entry : this->dom_tree_) {
        if (entry.second == super_source) {
            entry.second = boost::graph_traits<graph::Graph>::null_vertex();
        }
    }
    this->dom_tree_.erase(super_source);

    for (auto& entry : this->pdom_tree_) {
        if (entry.second == super_sink) {
            entry.second = boost::graph_traits<graph::Graph>::null_vertex();
        }
    }
    this->pdom_tree_.erase(super_sink);

    boost::clear_vertex(super_source, users_analysis.graph_);
    boost::clear_vertex(super_sink, users_analysis.graph_);
}

bool DominanceAnalysis::dominates(User& user1, User& user2) {
    auto dominator = this->dom_tree_.at(user2.vertex_);
    while (dominator != boost::graph_traits<graph::Graph>::null_vertex()) {
        if (dominator == user1.vertex_) {
            return true;
        }
        dominator = this->dom_tree_.at(dominator);
    }
    return false;
};

bool DominanceAnalysis::post_dominates(User& user1, User& user2) {
    auto dominator = this->pdom_tree_.at(user2.vertex_);
    while (dominator != boost::graph_traits<graph::Graph>::null_vertex()) {
        if (dominator == user1.vertex_) {
            return true;
        }
        dominator = this->pdom_tree_.at(dominator);
    }
    return false;
};

} // namespace analysis
} // namespace sdfg
