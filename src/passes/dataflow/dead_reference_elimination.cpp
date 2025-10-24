#include "sdfg/passes/dataflow/dead_reference_elimination.h"

#include "sdfg/analysis/users.h"

namespace sdfg {
namespace passes {

DeadReferenceElimination::DeadReferenceElimination()
    : Pass() {

      };

std::string DeadReferenceElimination::name() { return "DeadReferenceElimination"; };

bool DeadReferenceElimination::
    run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    bool applied = false;

    auto& sdfg = builder.subject();
    auto& users = analysis_manager.get<analysis::Users>();

    std::list<std::string> to_delete;
    for (auto& name : sdfg.containers()) {
        if (!sdfg.is_transient(name)) {
            continue;
        }
        auto& type = sdfg.type(name);
        if (!dynamic_cast<const types::Pointer*>(&type)) {
            continue;
        }
        if (type.storage_type().allocation() == types::StorageType::AllocationType::Managed ||
            type.storage_type().deallocation() == types::StorageType::AllocationType::Managed) {
            continue;
        }

        // Requirement: Pointer is only assigned
        auto reads = users.reads(name);
        auto writes = users.writes(name);
        if (reads.size() > 0 || writes.size() > 0) {
            continue;
        }
        auto views = users.views(name);
        auto moves = users.moves(name);
        if (views.size() > 0) {
            continue;
        }

        for (auto& move : moves) {
            auto access_node = dynamic_cast<data_flow::AccessNode*>(move->element());
            auto& graph = dynamic_cast<data_flow::DataFlowGraph&>(access_node->get_parent());
            auto& block = dynamic_cast<structured_control_flow::Block&>(*graph.get_parent());
            builder.clear_node(block, *access_node);
            applied = true;
        }

        to_delete.push_back(name);
    }
    for (auto& name : to_delete) {
        builder.remove_container(name);
    }

    return applied;
};

} // namespace passes
} // namespace sdfg
