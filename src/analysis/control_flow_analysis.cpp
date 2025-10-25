#include "sdfg/analysis/control_flow_analysis.h"

#include <cassert>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "sdfg/data_flow/memlet.h"
#include "sdfg/element.h"
#include "sdfg/graph/graph.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/symbolic/sets.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace analysis {
ControlFlowAnalysis::ControlFlowAnalysis(StructuredSDFG& sdfg)
    : Analysis(sdfg) {

      };

std::pair<graph::Vertex, graph::Vertex> ControlFlowAnalysis::traverse(structured_control_flow::ControlFlowNode& current
) {
    // Leaf nodes
    if (auto block_node = dynamic_cast<structured_control_flow::Block*>(&current)) {
        auto v = boost::add_vertex(graph_);
        nodes_[v] = &current;
        return {v, v};
    } else if (auto return_node = dynamic_cast<structured_control_flow::Return*>(&current)) {
        auto v = boost::add_vertex(graph_);
        nodes_[v] = &current;
        return {v, boost::graph_traits<graph::Graph>::null_vertex()};
    } else if (auto continue_node = dynamic_cast<structured_control_flow::Continue*>(&current)) {
        auto v = boost::add_vertex(graph_);
        nodes_[v] = &current;
        boost::add_edge(v, last_loop_, graph_);
        return {v, boost::graph_traits<graph::Graph>::null_vertex()};
    } else if (auto break_node = dynamic_cast<structured_control_flow::Break*>(&current)) {
        auto v = boost::add_vertex(graph_);
        nodes_[v] = &current;
        boost::add_edge(v, last_loop_, graph_);
        return {v, boost::graph_traits<graph::Graph>::null_vertex()};
    } else if (auto if_else_node = dynamic_cast<structured_control_flow::IfElse*>(&current)) {
        auto start_v = boost::add_vertex(graph_);
        nodes_[start_v] = &current;

        graph::Vertex end_v = boost::graph_traits<graph::Graph>::null_vertex();
        for (size_t i = 0; i < if_else_node->size(); i++) {
            auto [case_start, case_end] = this->traverse(if_else_node->at(i).first);
            if (case_start != boost::graph_traits<graph::Graph>::null_vertex()) {
                boost::add_edge(start_v, case_start, graph_);
            }
            if (case_end != boost::graph_traits<graph::Graph>::null_vertex()) {
                if (end_v == boost::graph_traits<graph::Graph>::null_vertex()) {
                    end_v = boost::add_vertex(graph_);
                    nodes_[end_v] = nullptr;
                }
                boost::add_edge(case_end, end_v, graph_);
            }
        }

        if (!if_else_node->is_complete()) {
            if (end_v == boost::graph_traits<graph::Graph>::null_vertex()) {
                end_v = boost::add_vertex(graph_);
                nodes_[end_v] = nullptr;
            }
            boost::add_edge(start_v, end_v, graph_);
        }

        return {start_v, end_v};
    } else if (auto while_loop = dynamic_cast<structured_control_flow::While*>(&current)) {
        auto header_v = boost::add_vertex(graph_);
        nodes_[header_v] = &current;

        auto previous_loop_ = last_loop_;
        last_loop_ = header_v;

        auto [body_start, body_end] = this->traverse(while_loop->root());
        if (body_start != boost::graph_traits<graph::Graph>::null_vertex()) {
            boost::add_edge(header_v, body_start, graph_);
        }
        if (body_end != boost::graph_traits<graph::Graph>::null_vertex()) {
            boost::add_edge(body_end, header_v, graph_);
        }

        last_loop_ = previous_loop_;

        return {header_v, header_v};
    } else if (auto structured_loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&current)) {
        auto header_v = boost::add_vertex(graph_);
        nodes_[header_v] = &current;

        auto previous_loop_ = last_loop_;
        last_loop_ = header_v;

        auto [body_start, body_end] = this->traverse(structured_loop->root());
        if (body_start != boost::graph_traits<graph::Graph>::null_vertex()) {
            boost::add_edge(header_v, body_start, graph_);
        }
        if (body_end != boost::graph_traits<graph::Graph>::null_vertex()) {
            boost::add_edge(body_end, header_v, graph_);
        }

        last_loop_ = previous_loop_;

        return {header_v, header_v};
    } else if (auto sequence_node = dynamic_cast<structured_control_flow::Sequence*>(&current)) {
        graph::Vertex seq_start = boost::graph_traits<graph::Graph>::null_vertex();
        graph::Vertex seq_end = boost::graph_traits<graph::Graph>::null_vertex();

        for (size_t i = 0; i < sequence_node->size(); i++) {
            auto& child = sequence_node->at(i).first;
            auto [child_start, child_end] = this->traverse(child);
            if (child_start != boost::graph_traits<graph::Graph>::null_vertex()) {
                if (seq_start == boost::graph_traits<graph::Graph>::null_vertex()) {
                    seq_start = child_start;
                }
                if (seq_end != boost::graph_traits<graph::Graph>::null_vertex()) {
                    boost::add_edge(seq_end, child_start, graph_);
                }
            }

            seq_end = child_end;
            if (seq_end == boost::graph_traits<graph::Graph>::null_vertex()) {
                break;
            }
        }

        return {seq_start, seq_end};
    } else {
        throw InvalidSDFGException("Unknown control flow node type encountered during control flow analysis.");
    }
};

void ControlFlowAnalysis::run(analysis::AnalysisManager& analysis_manager) {
    nodes_.clear();
    graph_.clear();
    dom_tree_.clear();
    pdom_tree_.clear();

    this->traverse(sdfg_.root());

    graph::Vertex entry_vertex = boost::graph_traits<graph::Graph>::null_vertex();
    for (auto v : boost::make_iterator_range(boost::vertices(graph_))) {
        if (boost::in_degree(v, graph_) == 0) {
            assert(entry_vertex == boost::graph_traits<graph::Graph>::null_vertex());
            entry_vertex = v;
        }
    }

    auto dom_tree = graph::dominator_tree(graph_, entry_vertex);
    for (const auto& [node, dom_node] : dom_tree) {
        if (nodes_.find(node) != nodes_.end() && nodes_.find(dom_node) != nodes_.end()) {
            dom_tree_[nodes_.at(node)] = nodes_.at(dom_node);
        }
    }

    auto pdom_tree = graph::post_dominator_tree(graph_);
    for (const auto& [node, pdom_node] : pdom_tree) {
        if (nodes_.find(node) != nodes_.end() && nodes_.find(pdom_node) != nodes_.end()) {
            pdom_tree_[nodes_.at(node)] = nodes_.at(pdom_node);
        }
    }
};

std::unordered_set<structured_control_flow::ControlFlowNode*> ControlFlowAnalysis::exits() const {
    std::unordered_set<structured_control_flow::ControlFlowNode*> exit_nodes;

    for (auto v : boost::make_iterator_range(boost::vertices(graph_))) {
        if (boost::out_degree(v, graph_) == 0) {
            assert(nodes_.find(v) != nodes_.end());
            exit_nodes.insert(nodes_.at(v));
        }
    }

    return exit_nodes;
}

const std::unordered_map<structured_control_flow::ControlFlowNode*, structured_control_flow::ControlFlowNode*>&
ControlFlowAnalysis::dom_tree() const {
    return dom_tree_;
}

const std::unordered_map<structured_control_flow::ControlFlowNode*, structured_control_flow::ControlFlowNode*>&
ControlFlowAnalysis::pdom_tree() const {
    return pdom_tree_;
}

bool ControlFlowAnalysis::
    dominates(structured_control_flow::ControlFlowNode& a, structured_control_flow::ControlFlowNode& b) const {
    auto it = dom_tree_.find(&b);
    while (it != dom_tree_.end()) {
        if (it->second == &a) {
            return true;
        }
        it = dom_tree_.find(it->second);
    }
    return false;
}

bool ControlFlowAnalysis::
    post_dominates(structured_control_flow::ControlFlowNode& a, structured_control_flow::ControlFlowNode& b) const {
    auto it = pdom_tree_.find(&b);
    while (it != pdom_tree_.end()) {
        if (it->second == &a) {
            return true;
        }
        it = pdom_tree_.find(it->second);
    }
    return false;
}

} // namespace analysis
} // namespace sdfg
