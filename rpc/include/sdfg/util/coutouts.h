#pragma once

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/deepcopy/structured_sdfg_deep_copy.h"
#include "sdfg/structured_sdfg.h"

namespace sdfg {
namespace util {

std::unique_ptr<StructuredSDFG> cutout(
    sdfg::builder::StructuredSDFGBuilder& builder,
    sdfg::analysis::AnalysisManager& analysis_manager,
    structured_control_flow::ControlFlowNode& node
) {
    auto& sdfg = builder.subject();
    builder::StructuredSDFGBuilder cutout_builder(
        sdfg.name() + "_" + std::to_string(node.element_id()), sdfg.type(), types::Scalar(types::PrimitiveType::Void)
    );

    // Register structure definitions
    for (auto& struct_name : sdfg.structures()) {
        auto& struct_def = sdfg.structure(struct_name);
        auto& struct_def_cutout = cutout_builder.add_structure(struct_def.name(), struct_def.is_packed());

        for (size_t i = 0; i < struct_def.num_members(); i++) {
            struct_def_cutout.add_member(struct_def.member_type(symbolic::integer(i)));
        }
    }

    // Register containers
    auto& users_analysis = analysis_manager.get<analysis::Users>();
    sdfg::analysis::UsersView node_users(users_analysis, node);
    for (auto& use : node_users.uses()) {
        std::string& container = use->container();
        if (cutout_builder.subject().exists(container)) {
            continue;
        }
        auto& type = builder.subject().type(container);
        cutout_builder.add_container(
            container,
            type,
            false, // Need to determine arguments of node via argument analysis
            builder.subject().is_external(container)
        );
    }

    // Copy nodes
    sdfg::deepcopy::StructuredSDFGDeepCopy deep_copy(cutout_builder, cutout_builder.subject().root(), node);
    deep_copy.copy();

    return cutout_builder.move();
}

} // namespace util
} // namespace sdfg
