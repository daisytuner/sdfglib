#pragma once

#include <boost/graph/detail/adjacency_list.hpp>
#include <cassert>
#include <cstddef>
#include <ranges>
#include <string>

#include "function.h"
#include "sdfg/control_flow/interstate_edge.h"
#include "sdfg/control_flow/state.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/data_flow/data_flow_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/element.h"
#include "sdfg/function.h"
#include "sdfg/graph/graph.h"
#include "sdfg/helpers/helpers.h"

namespace sdfg {

namespace builder {
class SDFGBuilder;
class StructuredSDFGBuilder;
} // namespace builder

class SDFG : public Function {
    friend class sdfg::builder::SDFGBuilder;
    friend class sdfg::builder::StructuredSDFGBuilder;

private:
    // Control-Flow Graph
    graph::Graph graph;
    std::unordered_map<graph::Vertex, std::unique_ptr<control_flow::State>, boost::hash<graph::Vertex>> states;
    std::unordered_map<graph::Edge, std::unique_ptr<control_flow::InterstateEdge>, boost::hash<graph::Edge>> edges;

    const control_flow::State* start_state;

public:
    SDFG(const std::string& name, FunctionType type);

    SDFG(const SDFG& sdfg) = delete;
    auto operator=(const SDFG&) -> SDFG& = delete;

    [[nodiscard]] auto debug_info() const -> const DebugInfo override;

    /***** Section: Graph *****/

    [[nodiscard]] auto states() const {
        return std::views::values(this->states) | std::views::transform(helpers::indirect<control_flow::State>) |
               std::views::transform(helpers::add_const<control_flow::State>);
    };

    [[nodiscard]] auto edges() const {
        return std::views::values(this->edges) |
               std::views::transform(helpers::indirect<control_flow::InterstateEdge>) |
               std::views::transform(helpers::add_const<control_flow::InterstateEdge>);
    };

    [[nodiscard]] auto in_edges(const control_flow::State& state) const {
        auto [eb, ee] = boost::in_edges(state.vertex(), this->graph);
        auto edges = std::ranges::subrange(eb, ee);

        // Convert Edge to const InterstateEdge&
        auto interstate_edges = std::views::transform(
                                    edges,
                                    [&lookup_table = this->edges](const graph::Edge& edge
                                    ) -> control_flow::InterstateEdge& { return *(lookup_table.find(edge)->second); }
                                ) |
                                std::views::transform(helpers::add_const<control_flow::InterstateEdge>);

        return interstate_edges;
    };

    [[nodiscard]] auto out_edges(const control_flow::State& state) const {
        auto [eb, ee] = boost::out_edges(state.vertex(), this->graph);
        auto edges = std::ranges::subrange(eb, ee);

        // Convert Edge to const InterstateEdge&
        auto interstate_edges = std::views::transform(
                                    edges,
                                    [&lookup_table = this->edges](const graph::Edge& edge
                                    ) -> control_flow::InterstateEdge& { return *(lookup_table.find(edge)->second); }
                                ) |
                                std::views::transform(helpers::add_const<control_flow::InterstateEdge>);

        return interstate_edges;
    };

    [[nodiscard]] auto start_state() const -> const control_flow::State&;

    [[nodiscard]] auto terminal_states() const {
        return this->states() |
               std::views::filter([this](const control_flow::State& state) { return this->out_degree(state) == 0; });
    };

    [[nodiscard]] auto in_degree(const control_flow::State& state) const -> size_t;

    [[nodiscard]] auto out_degree(const control_flow::State& state) const -> size_t;

    [[nodiscard]] auto is_adjacent(const control_flow::State& src, const control_flow::State& dst) const -> bool;

    [[nodiscard]] auto edge(const control_flow::State& src, const control_flow::State& dst) const
        -> const control_flow::InterstateEdge&;

    [[nodiscard]] auto dominator_tree() const
        -> std::unordered_map<const control_flow::State*, const control_flow::State*>;

    [[nodiscard]] auto post_dominator_tree() const
        -> std::unordered_map<const control_flow::State*, const control_flow::State*>;

    [[nodiscard]] auto back_edges() const -> std::list<const control_flow::InterstateEdge*>;

    [[nodiscard]] auto all_simple_paths(const control_flow::State& src, const control_flow::State& dst) const
        -> std::list<std::list<const control_flow::InterstateEdge*>>;
};

} // namespace sdfg
