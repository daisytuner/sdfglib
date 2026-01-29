#pragma once

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/arguments_analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/cutouts/cutout_serializer.h"
#include "sdfg/structured_sdfg.h"

namespace sdfg {
namespace util {

std::unique_ptr<StructuredSDFG> cutout(
    sdfg::builder::StructuredSDFGBuilder& builder,
    sdfg::analysis::AnalysisManager& analysis_manager,
    structured_control_flow::ControlFlowNode& node
) {
    structured_control_flow::Sequence* sequence;
    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();

    sequence = dynamic_cast<structured_control_flow::Sequence*>(&node);
    if (!sequence) {
        structured_control_flow::Sequence* parent_scope = nullptr;
        structured_control_flow::Sequence* new_scope = nullptr;
        int index = -1;
        parent_scope = static_cast<structured_control_flow::Sequence*>(scope_analysis.parent_scope(&node));
        index = parent_scope->index(node);

        new_scope = &builder.add_sequence_before(*parent_scope, node, {}, {});
        builder.move_child(*parent_scope, index + 1, *new_scope);
        analysis_manager.invalidate_all();
        sequence = new_scope;
    }
    auto& sdfg = builder.subject();

    serializer::CutoutSerializer serializer;
    nlohmann::json cutout_json = serializer.serialize(sdfg, &analysis_manager, sequence);
    auto cutout_sdfg = serializer.deserialize(cutout_json);

    return cutout_sdfg;
}

} // namespace util
} // namespace sdfg
