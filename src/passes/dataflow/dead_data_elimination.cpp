#include "sdfg/passes/dataflow/dead_data_elimination.h"

#include "sdfg/analysis/data_dependency_analysis.h"

namespace sdfg {
namespace passes {

DeadDataElimination::DeadDataElimination() : Pass() {};

std::string DeadDataElimination::name() { return "DeadDataElimination"; };

bool DeadDataElimination::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    bool applied = false;

    auto& sdfg = builder.subject();
    auto& users = analysis_manager.get<analysis::Users>();
    auto& data_dependency_analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();

    // Eliminate dead code, i.e., never read
    std::unordered_set<std::string> dead;
    for (auto& name : sdfg.containers()) {
        if (!sdfg.is_transient(name)) {
            continue;
        }
        if (!dynamic_cast<const types::Scalar*>(&sdfg.type(name))) {
            continue;
        }

        if (!users.views(name).empty() || !users.moves(name).empty()) {
            continue;
        }
        if (users.reads(name).empty() && users.writes(name).empty()) {
            dead.insert(name);
            applied = true;
            continue;
        }

        // Writes without reads
        auto raws = data_dependency_analysis.definitions(name);
        for (auto set : raws) {
            if (set.second.size() > 0) {
                continue;
            }
            if (data_dependency_analysis.is_undefined_user(*set.first)) {
                continue;
            }

            auto write = set.first;
            if (auto transition = dynamic_cast<structured_control_flow::Transition*>(write->element())) {
                transition->assignments().erase(symbolic::symbol(name));
                applied = true;
            } else if (auto access_node = dynamic_cast<data_flow::AccessNode*>(write->element())) {
                auto& graph = access_node->get_parent();

                auto& src = (*graph.in_edges(*access_node).begin()).src();
                if (auto tasklet = dynamic_cast<data_flow::Tasklet*>(&src)) {
                    auto& block = dynamic_cast<structured_control_flow::Block&>(*graph.get_parent());
                    builder.clear_node(block, *tasklet);
                    applied = true;
                } else if (auto library_node = dynamic_cast<data_flow::LibraryNode*>(&src)) {
                    if (!library_node->side_effect()) {
                        auto& block = dynamic_cast<structured_control_flow::Block&>(*graph.get_parent());
                        builder.clear_node(block, *library_node);
                        applied = true;
                    }
                }
            }
        }
    }

    for (auto& name : dead) {
        builder.remove_container(name);
    }

    return applied;
};

} // namespace passes
} // namespace sdfg
