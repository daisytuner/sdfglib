#pragma once

#include <boost/graph/graphviz.hpp>
#include <boost/lexical_cast.hpp>
#include <cassert>
#include <functional>
#include <iostream>
#include <list>
#include <memory>
#include <ranges>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "function.h"
#include "sdfg/control_flow/interstate_edge.h"
#include "sdfg/control_flow/state.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/data_flow/data_flow_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/function.h"
#include "sdfg/graph/graph.h"
#include "sdfg/helpers/helpers.h"
#include "sdfg/types/array.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/structure.h"
#include "sdfg/types/type.h"

namespace sdfg {

namespace builder {
class SDFGBuilder;
class StructuredSDFGBuilder;
} // namespace builder

// Analysis structures
namespace analysis {
enum class LoopExitKind { Continue, Break, Return };

struct LoopExit {
    const control_flow::State* from;
    const control_flow::State* to;
    LoopExitKind kind;
};

struct NaturalLoop {
    const control_flow::State* header;
    std::vector<const control_flow::State*> latches;
    std::unordered_set<const control_flow::State*> body;
    std::unordered_set<const control_flow::State*> exits;
};

struct SCCInfo {
    std::unordered_map<const control_flow::State*, size_t> component_of;
    size_t num_components{0};
    std::unordered_set<size_t> irreducible_components;
    std::unordered_map<size_t, std::unordered_set<const control_flow::State*>> component_states;
};
} // namespace analysis

class SDFG : public Function {
    friend class sdfg::builder::SDFGBuilder;
    friend class sdfg::builder::StructuredSDFGBuilder;

private:
    // Control-Flow Graph
    graph::Graph graph_;
    std::unordered_map<graph::Vertex, std::unique_ptr<control_flow::State>, boost::hash<graph::Vertex>> states_;
    std::unordered_map<graph::Edge, std::unique_ptr<control_flow::InterstateEdge>, boost::hash<graph::Edge>> edges_;

    const control_flow::State* start_state_;

    // Cached analyses (lazy)
    mutable std::optional<std::unordered_map<const control_flow::State*, const control_flow::State*>> dom_cache_;
    mutable std::optional<std::unordered_map<const control_flow::State*, const control_flow::State*>> pdom_cache_;
    mutable std::optional<std::unordered_map<const control_flow::State*, std::unordered_set<const control_flow::State*>>>
        df_cache_;
    mutable std::optional<analysis::SCCInfo> scc_cache_;
    mutable std::optional<std::vector<analysis::NaturalLoop>> natural_loops_cache_;
    mutable std::optional<std::unordered_map<const control_flow::State*, std::vector<analysis::LoopExit>>>
        loop_exit_cache_;

public:
    SDFG(const std::string& name, FunctionType type);
    SDFG(const std::string& name, FunctionType type, const types::IType& return_type);

    SDFG(const SDFG& sdfg) = delete;
    SDFG& operator=(const SDFG&) = delete;

    void validate() const override;

    auto states() const {
        return std::views::values(this->states_) | std::views::transform(helpers::indirect<control_flow::State>) |
               std::views::transform(helpers::add_const<control_flow::State>);
    };

    auto edges() const {
        return std::views::values(this->edges_) |
               std::views::transform(helpers::indirect<control_flow::InterstateEdge>) |
               std::views::transform(helpers::add_const<control_flow::InterstateEdge>);
    };

    auto in_edges(const control_flow::State& state) const {
        auto [eb, ee] = boost::in_edges(state.vertex(), this->graph_);
        auto edges = std::ranges::subrange(eb, ee);

        // Convert Edge to const InterstateEdge&
        auto interstate_edges = std::views::transform(
                                    edges,
                                    [&lookup_table = this->edges_](const graph::Edge& edge
                                    ) -> control_flow::InterstateEdge& { return *(lookup_table.find(edge)->second); }
                                ) |
                                std::views::transform(helpers::add_const<control_flow::InterstateEdge>);

        return interstate_edges;
    };

    auto out_edges(const control_flow::State& state) const {
        auto [eb, ee] = boost::out_edges(state.vertex(), this->graph_);
        auto edges = std::ranges::subrange(eb, ee);

        // Convert Edge to const InterstateEdge&
        auto interstate_edges = std::views::transform(
                                    edges,
                                    [&lookup_table = this->edges_](const graph::Edge& edge
                                    ) -> control_flow::InterstateEdge& { return *(lookup_table.find(edge)->second); }
                                ) |
                                std::views::transform(helpers::add_const<control_flow::InterstateEdge>);

        return interstate_edges;
    };

    const control_flow::State& start_state() const;

    auto terminal_states() const {
        return this->states() |
               std::views::filter([this](const control_flow::State& state) { return this->out_degree(state) == 0; });
    };

    size_t in_degree(const control_flow::State& state) const;

    size_t out_degree(const control_flow::State& state) const;

    bool is_adjacent(const control_flow::State& src, const control_flow::State& dst) const;

    const control_flow::InterstateEdge& edge(const control_flow::State& src, const control_flow::State& dst) const;

    std::list<const control_flow::InterstateEdge*> back_edges() const;

    std::unordered_map<const control_flow::State*, const control_flow::State*> dominator_tree() const;

    std::unordered_map<const control_flow::State*, const control_flow::State*> post_dominator_tree();

    const std::unordered_map<const control_flow::State*, std::unordered_set<const control_flow::State*>>&
    dominance_frontiers();

    const analysis::SCCInfo& scc_info();

    const std::vector<analysis::NaturalLoop>& natural_loops();

    const std::unordered_map<const control_flow::State*, std::vector<analysis::LoopExit>>& loop_exits();
};

} // namespace sdfg
