#include "sdfg/passes/canonicalize_cfg.h"

namespace sdfg {
namespace passes {

std::unordered_set<const control_flow::State*> CanonicalizeCFG::determine_loop_nodes(
    const SDFG& sdfg, const control_flow::State& start, const control_flow::State& end) const {
    std::unordered_set<const control_flow::State*> nodes;
    std::unordered_set<const control_flow::State*> visited;
    std::list<const control_flow::State*> queue = {&start};
    while (!queue.empty()) {
        auto curr = queue.front();
        queue.pop_front();
        nodes.insert(curr);

        if (visited.find(curr) != visited.end()) {
            continue;
        }
        visited.insert(curr);

        if (curr == &end) {
            continue;
        }

        for (auto& iedge : sdfg.in_edges(*curr)) {
            if (visited.find(&iedge.src()) != visited.end()) {
                continue;
            }
            queue.push_back(&iedge.src());
        }
    }

    return nodes;
};

std::string CanonicalizeCFG::name() { return "CanonicalizeCFG"; };

bool CanonicalizeCFG::run_pass(builder::SDFGBuilder& builder) {
    bool applied = false;

    auto& sdfg = builder.subject();

    //  Single Exit SDFG
    // guarantees that there is a single exit state
    std::unordered_set<const control_flow::State*> terminal_states;
    for (auto& state : sdfg.states()) {
        if (sdfg.out_degree(state) == 0) {
            terminal_states.insert(&state);
        }
    }
    if (terminal_states.size() > 1) {
        auto& term = builder.add_state();
        for (auto state : terminal_states) {
            builder.add_edge(*state, term);
        }

        applied = true;
    }

    // Single exit loops
    // guarantees that there is a single exit per loop
    auto bedges = sdfg.back_edges();
    for (auto back_edge : bedges) {
        auto loop_states = determine_loop_nodes(sdfg, back_edge->src(), back_edge->dst());

        // Determine loop exits
        std::unordered_set<const control_flow::State*> exit_states;
        std::unordered_map<const control_flow::State*,
                           std::unordered_set<const control_flow::InterstateEdge*>>
            exit_edges;
        for (auto& node : loop_states) {
            for (auto& edge : sdfg.out_edges(*node)) {
                auto exit_state = &edge.dst();
                if (loop_states.find(exit_state) == loop_states.end()) {
                    exit_states.insert(exit_state);
                    if (exit_edges.find(exit_state) == exit_edges.end()) {
                        exit_edges.insert({exit_state, {}});
                    }
                    exit_edges[exit_state].insert(&edge);
                }
            }
        }
        // Merge exit states
        if (exit_states.size() > 1) {
            auto& merged_state = builder.add_state(false);

            auto sym_name = builder.find_new_name("daisy_jump_");
            builder.add_container(sym_name, types::Scalar(types::PrimitiveType::UInt64));
            auto sym = symbolic::symbol(sym_name);

            size_t i = 0;
            for (auto exit_state : exit_states) {
                for (auto iedge : exit_edges[exit_state]) {
                    auto assignments = iedge->assignments();
                    assignments[sym] = symbolic::integer(i);
                    builder.add_edge(iedge->src(), merged_state, assignments, iedge->condition());
                }

                builder.add_edge(merged_state, *exit_state,
                                 symbolic::Eq(sym, symbolic::integer(i)));

                for (auto iedge : exit_edges[exit_state]) {
                    builder.remove_edge(*iedge);
                }

                i++;
            }

            applied = true;
            break;
        }
    }

    return applied;
};

}  // namespace passes
}  // namespace sdfg
