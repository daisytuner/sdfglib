#include "sdfg/passes/dataflow/tasklet_fusion.h"

#include <unordered_set>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/element.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/types/type.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

bool TaskletFusion::container_allowed_accesses(const std::string& container, const Element* allowed_read_and_writes) {
    for (auto* user : this->users_analysis_.uses(container)) {
        if ((user->use() == analysis::Use::READ || user->use() == analysis::Use::WRITE) &&
            user->element() != allowed_read_and_writes) {
            return false;
        }
    }
    return true;
}

TaskletFusion::TaskletFusion(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
    : visitor::NonStoppingStructuredSDFGVisitor(builder, analysis_manager),
      users_analysis_(analysis_manager.get<analysis::Users>()) {};

bool TaskletFusion::accept(structured_control_flow::Block& block) {
    bool applied = false;
    auto& dfg = block.dataflow();

    // Fuse "input" assignments:
    for (auto* access_node : dfg.data_nodes()) {
        // Require exactly one in edge and at leat one out edge
        if (dfg.in_degree(*access_node) != 1 || dfg.out_degree(*access_node) == 0) {
            continue;
        }

        // The in edge must provide a scalar
        auto& iedge = *dfg.in_edges(*access_node).begin();
        if (iedge.base_type().type_id() != types::TypeID::Scalar) {
            continue;
        }

        // The source of the in edge must be a tasklet with an assignment
        auto* assign_tasklet = dynamic_cast<data_flow::Tasklet*>(&iedge.src());
        if (!assign_tasklet || assign_tasklet->code() != data_flow::TaskletCode::assign ||
            dfg.in_degree(*assign_tasklet) != 1) {
            continue;
        }
        auto& assign_tasklet_iedge = *dfg.in_edges(*assign_tasklet).begin();

        // All out edges must provide scalars and their destination must be a tasklet
        std::unordered_set<data_flow::Memlet*> oedges;
        for (auto& oedge : dfg.out_edges(*access_node)) {
            if (oedge.base_type().type_id() == types::TypeID::Scalar &&
                dynamic_cast<data_flow::Tasklet*>(&oedge.dst())) {
                oedges.insert(&oedge);
            }
        }
        if (oedges.size() != dfg.out_degree(*access_node)) {
            continue;
        }

        // Container is only read and written in this access node (SSA)
        if (!this->container_allowed_accesses(access_node->data(), access_node)) {
            continue;
        }

        // Replace each out edge with an edge from the input access node from the assignment tasklet to the out edges's
        // tasklet
        for (auto* oedge : oedges) {
            auto& tasklet = dynamic_cast<data_flow::Tasklet&>(oedge->dst());
            DebugInfo debug_info = DebugInfo::merge(
                assign_tasklet_iedge.debug_info(),
                DebugInfo::merge(
                    assign_tasklet->debug_info(),
                    DebugInfo::merge(iedge.debug_info(), DebugInfo::merge(access_node->debug_info(), oedge->debug_info()))
                )
            );
            this->builder_.add_computational_memlet(
                block,
                dynamic_cast<data_flow::AccessNode&>(assign_tasklet_iedge.src()),
                tasklet,
                oedge->dst_conn(),
                assign_tasklet_iedge.subset(),
                debug_info
            );
            this->builder_.remove_memlet(block, *oedge);
        }

        // Remove the obsolete memlets
        this->builder_.remove_memlet(block, assign_tasklet_iedge);
        this->builder_.remove_memlet(block, iedge);

        // Remove the obsolete nodes
        this->builder_.remove_node(block, *assign_tasklet);
        this->builder_.remove_node(block, *access_node);

        applied = true;
    }

    // Fuse "output" assignments:
    for (auto* access_node : dfg.data_nodes()) {
        // Require exactly one in and one out edge
        if (dfg.in_degree(*access_node) != 1 || dfg.out_degree(*access_node) != 1) {
            continue;
        }

        // The in edge must provide a scalar
        auto& iedge = *dfg.in_edges(*access_node).begin();
        if (iedge.base_type().type_id() != types::TypeID::Scalar) {
            continue;
        }

        // The out edge must provide a scalar
        auto& oedge = *dfg.out_edges(*access_node).begin();
        if (oedge.base_type().type_id() != types::TypeID::Scalar) {
            continue;
        }

        // The source of the in edge must be a tasklet
        auto* tasklet = dynamic_cast<data_flow::Tasklet*>(&iedge.src());
        if (!tasklet) {
            continue;
        }

        // The destination of the out edge must be a tasklet with an assignment
        auto* assign_tasklet = dynamic_cast<data_flow::Tasklet*>(&oedge.dst());
        if (!assign_tasklet || assign_tasklet->code() != data_flow::TaskletCode::assign ||
            dfg.out_degree(*assign_tasklet) != 1) {
            continue;
        }
        auto& assign_tasklet_oedge = *dfg.out_edges(*assign_tasklet).begin();

        // Container is only read and written in this access node (SSA)
        if (!this->container_allowed_accesses(access_node->data(), access_node)) {
            continue;
        }

        // Replace the in edge with an edge from the in edge's tasklet to the output access node
        DebugInfo debug_info = DebugInfo::merge(
            iedge.debug_info(),
            DebugInfo::merge(
                access_node->debug_info(),
                DebugInfo::merge(
                    oedge.debug_info(),
                    DebugInfo::merge(assign_tasklet->debug_info(), assign_tasklet_oedge.debug_info())
                )
            )
        );
        this->builder_.add_computational_memlet(
            block,
            *tasklet,
            iedge.src_conn(),
            dynamic_cast<data_flow::AccessNode&>(assign_tasklet_oedge.dst()),
            assign_tasklet_oedge.subset(),
            debug_info
        );
        this->builder_.remove_memlet(block, iedge);

        // Remove obsolete memlets
        this->builder_.remove_memlet(block, oedge);
        this->builder_.remove_memlet(block, assign_tasklet_oedge);

        // Remove obsolete nodes
        this->builder_.remove_node(block, *access_node);
        this->builder_.remove_node(block, *assign_tasklet);

        applied = true;
    }

    return applied;
}

} // namespace passes
} // namespace sdfg
