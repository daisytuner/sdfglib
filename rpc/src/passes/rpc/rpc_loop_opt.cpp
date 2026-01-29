#include "sdfg/passes/rpc/rpc_loop_opt.h"

#include "sdfg/transformations/rpc_node_transform.h"

namespace sdfg {
namespace passes {
namespace rpc {

RpcLoopOpt::RpcLoopOpt(rpc::RpcContext& rpc_context, std::string target, std::string category, bool print_steps)
    : LoopScheduler(), rpc_context_(rpc_context), target_(std::move(target)), category_(std::move(category)),
      print_steps_(print_steps) {}

scheduler::SchedulerAction RpcLoopOpt::schedule(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::StructuredLoop& loop,
    const scheduler::SchedulerLoopInfo& loop_info
) {
    // Apply transfer tuning to the loop
    transformations::RPCNodeTransform rpc_transform(loop, target_, category_, rpc_context_, print_steps_);

    if (rpc_transform.can_be_applied(builder, analysis_manager)) {
        rpc_transform.apply(builder, analysis_manager);
        return scheduler::NEXT;
    }

    return scheduler::NEXT;
}

scheduler::SchedulerAction RpcLoopOpt::schedule(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::While& loop,
    const scheduler::SchedulerLoopInfo& loop_info
) {
    // Apply transfer tuning to the loop
    transformations::RPCNodeTransform rpc_transform(loop, target_, category_, rpc_context_, print_steps_);
    if (rpc_transform.can_be_applied(builder, analysis_manager)) {
        rpc_transform.apply(builder, analysis_manager);
        return scheduler::NEXT;
    }

    return scheduler::NEXT;
}

} // namespace rpc
} // namespace passes
} // namespace sdfg
