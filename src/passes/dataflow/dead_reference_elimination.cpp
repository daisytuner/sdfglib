#include "sdfg/passes/dataflow/dead_reference_elimination.h"

namespace sdfg {
namespace passes {

DeadReferenceElimination::DeadReferenceElimination()
    : Pass(){

      };

std::string DeadReferenceElimination::name() { return "DeadReferenceElimination"; };

bool DeadReferenceElimination::run_pass(builder::StructuredSDFGBuilder& builder,
                                        analysis::AnalysisManager& analysis_manager) {
    bool applied = false;

    auto& sdfg = builder.subject();
    auto& users = analysis_manager.get<analysis::Users>();

    std::list<std::string> containers_copy;
    for (auto& name : sdfg.containers()) {
        if (!sdfg.is_transient(name)) {
            continue;
        }
        if (!dynamic_cast<const types::Pointer*>(&sdfg.type(name))) {
            continue;
        }
        containers_copy.push_back(name);
    }

    for (auto& name : containers_copy) {
        auto moves = users.moves(name);
        auto uses = users.uses(name);
        if (uses.size() != moves.size()) {
            continue;
        }

        bool assigned = false;
        for (auto& move : moves) {
            auto access_node = dynamic_cast<data_flow::AccessNode*>(move->element());
            auto& graph = dynamic_cast<data_flow::DataFlowGraph&>(access_node->get_parent());
            auto& edge = *graph.in_edges(*access_node).begin();
            if (edge.dst_conn() == "void") {
                assigned = true;
                break;
            }
        }
        if (assigned) {
            continue;
        }

        for (auto& move : moves) {
            auto access_node = dynamic_cast<data_flow::AccessNode*>(move->element());
            auto& graph = dynamic_cast<data_flow::DataFlowGraph&>(access_node->get_parent());
            auto& edge = *graph.in_edges(*access_node).begin();
            auto& block = dynamic_cast<structured_control_flow::Block&>(graph.get_parent());
            builder.clear_node(block, *access_node);
            applied = true;
        }

        builder.remove_container(name);
    }

    return applied;
};

}  // namespace passes
}  // namespace sdfg
