#include "sdfg/passes/canonicalize_cfg.h"

namespace sdfg {
namespace passes {

std::unordered_set<const control_flow::State*> CanonicalizeCFG::
    determine_loop_nodes(const SDFG& sdfg, const control_flow::State& start, const control_flow::State& end) const {
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
    return applied;
};

} // namespace passes
} // namespace sdfg
