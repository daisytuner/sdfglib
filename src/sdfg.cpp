#include "sdfg/sdfg.h"

#include "sdfg/codegen/language_extensions/c_language_extension.h"
#include "sdfg/element.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {

SDFG::SDFG(const std::string& name, FunctionType type, const types::IType& return_type)
    : Function(name, type, return_type), start_state_(nullptr) {};

SDFG::SDFG(const std::string& name, FunctionType type) : SDFG(name, type, types::Scalar(types::PrimitiveType::Void)) {};

void SDFG::validate() const {
    // Call parent validate
    Function::validate();

    // Validate states and edges
    for (auto& state : this->states()) {
        state.validate(*this);

        if (dynamic_cast<const control_flow::ReturnState*>(&state)) {
            if (this->out_degree(state) != 0) {
                throw InvalidSDFGException("ReturnState cannot have outgoing edges");
            }
        }
    }
    for (auto& edge : this->edges()) {
        edge.validate(*this);
    }

    for (auto& term : this->terminal_states()) {
        if (!dynamic_cast<const control_flow::ReturnState*>(&term)) {
            throw InvalidSDFGException("Terminal state is not a valid State");
        }
    }
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

const control_flow::InterstateEdge& SDFG::edge(const control_flow::State& src, const control_flow::State& dst) const {
    auto e = boost::edge(src.vertex(), dst.vertex(), this->graph_);
    if (!e.second) {
        throw InvalidSDFGException("Edge does not exist");
    }
    return *this->edges_.at(e.first);
};

const control_flow::State& SDFG::start_state() const {
    if (this->start_state_ == nullptr) {
        throw InvalidSDFGException("Start state not set");
    }
    return *this->start_state_;
};

std::list<const control_flow::InterstateEdge*> SDFG::back_edges() const {
    std::list<const control_flow::InterstateEdge*> bedges;
    for (const auto& edge : graph::back_edges(this->graph_, this->start_state_->vertex())) {
        bedges.push_back(this->edges_.find(edge)->second.get());
    }

    return bedges;
};

std::unordered_map<const control_flow::State*, const control_flow::State*> SDFG::dominator_tree() const {
    if (this->dom_cache_.has_value()) {
        return *this->dom_cache_;
    }

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

    this->dom_cache_ = std::move(dom_tree);
    return *this->dom_cache_;
};

std::unordered_map<const control_flow::State*, const control_flow::State*> SDFG::post_dominator_tree() {
    if (this->pdom_cache_.has_value()) {
        return *this->pdom_cache_;
    }

    auto pdom_tree_ = graph::post_dominator_tree(this->graph_);

    std::unordered_map<const control_flow::State*, const control_flow::State*> pdom_tree;
    for (auto& entry : pdom_tree_) {
        if (this->states_.find(entry.first) == this->states_.end()) {
            // This is the synthetic super-terminal, skip it
            continue;
        }

        control_flow::State* first = this->states_.at(entry.first).get();
        control_flow::State* second = nullptr;
        if (entry.second != boost::graph_traits<graph::Graph>::null_vertex() &&
            this->states_.find(entry.second) != this->states_.end()) {
            second = this->states_.at(entry.second).get();
        }
        pdom_tree.insert({first, second});
    }

    this->pdom_cache_ = std::move(pdom_tree);
    return *this->pdom_cache_;
};

const std::unordered_map<const control_flow::State*, std::unordered_set<const control_flow::State*>>& SDFG::
    dominance_frontiers() {
    if (this->df_cache_.has_value()) {
        return *this->df_cache_;
    }

    // Build idom mapping (State* -> State*) and reverse to vertex mapping
    auto dom_tree = this->dominator_tree();
    std::unordered_map<graph::Vertex, graph::Vertex> idom_vertex_map;
    for (const auto& [state, parent] : dom_tree) {
        graph::Vertex v = state->vertex();
        graph::Vertex p = parent ? parent->vertex() : boost::graph_traits<graph::Graph>::null_vertex();
        idom_vertex_map.emplace(v, p);
    }

    auto df_vertex = graph::dominance_frontiers(this->graph_, idom_vertex_map);
    std::unordered_map<const control_flow::State*, std::unordered_set<const control_flow::State*>> df_state;
    for (const auto& [v, frontier_set] : df_vertex) {
        if (this->states_.find(v) == this->states_.end()) continue; // skip synthetic
        const control_flow::State* s = this->states_.at(v).get();
        std::unordered_set<const control_flow::State*> fset;
        for (auto fv : frontier_set) {
            if (this->states_.find(fv) == this->states_.end()) continue;
            fset.insert(this->states_.at(fv).get());
        }
        df_state.emplace(s, std::move(fset));
    }

    this->df_cache_ = std::move(df_state);
    return *this->df_cache_;
};

const analysis::SCCInfo& SDFG::scc_info() {
    if (this->scc_cache_.has_value()) {
        return *this->scc_cache_;
    }

    auto scc_ = graph::classify_sccs_irreducible(this->graph_, this->start_state_->vertex());

    // Convert vertex-based SCCInfo to State*-based SCCInfo
    analysis::SCCInfo converted_scc;
    for (const auto& [v, comp_id] : scc_.component_of) {
        const control_flow::State* s = this->states_.at(v).get();
        converted_scc.component_of.emplace(s, comp_id);
    }
    converted_scc.num_components = scc_.num_components;
    converted_scc.irreducible_components = scc_.irreducible_components;
    for (const auto& [comp_id, vertices] : scc_.component_vertices) {
        std::unordered_set<const control_flow::State*> states;
        for (const auto& v : vertices) {
            states.insert(this->states_.at(v).get());
        }
        converted_scc.component_states.emplace(comp_id, std::move(states));
    }

    this->scc_cache_ = std::move(converted_scc);
    return *this->scc_cache_;
};

const std::vector<analysis::NaturalLoop>& SDFG::natural_loops() {
    if (this->natural_loops_cache_.has_value()) {
        return *this->natural_loops_cache_;
    }

    auto natural_loops_ = graph::natural_loops(this->graph_, this->start_state_->vertex());

    std::vector<analysis::NaturalLoop> converted_loops;
    for (auto& loop : natural_loops_) {
        analysis::NaturalLoop loop_converted;
        loop_converted.header = this->states_.at(loop.header).get();
        for (auto latch_v : loop.latches) {
            loop_converted.latches.push_back(this->states_.at(latch_v).get());
        }
        for (auto body_v : loop.body) {
            loop_converted.body.insert(this->states_.at(body_v).get());
        }
        for (auto exit_v : loop.exits) {
            loop_converted.exits.insert(this->states_.at(exit_v).get());
        }
        converted_loops.push_back(std::move(loop_converted));
    }

    this->natural_loops_cache_ = std::move(converted_loops);
    return *this->natural_loops_cache_;
};

const std::unordered_map<const control_flow::State*, std::vector<analysis::LoopExit>>& SDFG::loop_exits() {
    if (this->loop_exit_cache_.has_value()) {
        return *this->loop_exit_cache_;
    }

    auto pdom_tree = this->post_dominator_tree();
    auto loops = this->natural_loops();

    // Collect all headers for nested discrimination
    std::unordered_set<const control_flow::State*> all_headers;
    for (auto& loop : loops) {
        all_headers.insert(loop.header);
    }

    std::unordered_map<const control_flow::State*, std::vector<analysis::LoopExit>> result;
    struct ExitKey {
        const control_flow::State* f;
        const control_flow::State* t;
        analysis::LoopExitKind k;
    };
    struct ExitKeyHash {
        size_t operator()(const ExitKey& ek) const {
            return std::hash<const control_flow::State*>()(ek.f) ^
                   (std::hash<const control_flow::State*>()(ek.t) << 1) ^ (int) ek.k;
        }
    };
    struct ExitKeyEq {
        bool operator()(const ExitKey& a, const ExitKey& b) const { return a.f == b.f && a.t == b.t && a.k == b.k; }
    };

    for (const auto& loop : loops) {
        const control_flow::State* header = loop.header;
        std::vector<analysis::LoopExit> exits;
        std::unordered_set<ExitKey, ExitKeyHash, ExitKeyEq> seen;

        // Continue edges: latch -> header.
        // A latch that is also a loop header counts only if it is the current header itself.
        for (auto latch : loop.latches) {
            if (!this->is_adjacent(*latch, *header)) {
                continue;
            }

            if (latch != header && all_headers.count(latch)) {
                continue; // skip inner loop header acting as latch
            }

            ExitKey key{latch, loop.header, analysis::LoopExitKind::Continue};
            if (!seen.count(key)) {
                exits.push_back(analysis::LoopExit{latch, header, analysis::LoopExitKind::Continue});
                seen.insert(key);
            }
        }

        for (auto from_state : loop.body) {
            for (auto& oedge : this->out_edges(*from_state)) {
                auto to_state = &oedge.dst();
                if (loop.body.find(to_state) != loop.body.end()) {
                    continue;
                }

                bool is_return = dynamic_cast<const control_flow::ReturnState*>(to_state) != nullptr;

                // Skip exits originating from inner loop headers for outer loop (unless they are latches and edge is
                // continue already handled). from_state != header && header set contains from_state.
                if (from_state != header && all_headers.count(from_state)) {
                    // If this edge was latch->header it is already counted, other edges from inner header are ignored
                    // for outer classification.
                    continue;
                }

                // For this loop: do not treat transitions into another loop header as breaks except when from_state ==
                // header (inner loop breaking to outer header) or is_return.
                if (!is_return && to_state != header && all_headers.count(to_state) && from_state != header) {
                    continue;
                }

                // If non-latch body node directly goes to header (to_v == loop.header), that's internal reconnection,
                // skip (continue edges already handled).
                bool is_latch = std::find(loop.latches.begin(), loop.latches.end(), from_state) != loop.latches.end();
                if (!is_return && to_state == header && !is_latch) {
                    continue;
                }

                analysis::LoopExitKind kind = is_return ? analysis::LoopExitKind::Return
                                                        : analysis::LoopExitKind::Break;
                ExitKey key{from_state, to_state, kind};
                if (!seen.count(key)) {
                    exits.push_back(analysis::LoopExit{from_state, to_state, kind});
                    seen.insert(key);
                }
            }
        }

        result.emplace(header, std::move(exits));
    }

    this->loop_exit_cache_ = std::move(result);
    return *this->loop_exit_cache_;
};

} // namespace sdfg
