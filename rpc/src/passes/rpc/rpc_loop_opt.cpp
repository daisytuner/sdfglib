#include "sdfg/passes/rpc/rpc_loop_opt.h"

#include "sdfg/transformations/rpc_node_transform.h"

namespace sdfg {
namespace passes {
namespace scheduler {

RpcLoopOpt::RpcLoopOpt(rpc::RpcContext& rpc_context) : LoopScheduler(), rpc_context_(rpc_context) {}

SchedulerAction RpcLoopOpt::schedule(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::StructuredLoop& loop,
    const SchedulerLoopInfo& loop_info
) {
    // Apply transfer tuning to the loop
    transformations::RPCNodeTransform rpc_transform(loop, "sequential", "server", rpc_context_);

    if (rpc_transform.can_be_applied(builder, analysis_manager)) {
        rpc_transform.apply(builder, analysis_manager);
        return NEXT;
    }

    return NEXT;
}

SchedulerAction RpcLoopOpt::schedule(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::While& loop,
    const SchedulerLoopInfo& loop_info
) {
    // Apply transfer tuning to the loop
    transformations::RPCNodeTransform rpc_transform(loop, "sequential", "server", rpc_context_);
    if (rpc_transform.can_be_applied(builder, analysis_manager)) {
        rpc_transform.apply(builder, analysis_manager);
        return NEXT;
    }

    return NEXT;
}

} // namespace scheduler
} // namespace passes
} // namespace sdfg
