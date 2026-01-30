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
    if (loop_info.loop_info.loopnest_index == -1 || loop_info.loop_info.has_side_effects) {
        return scheduler::NEXT;
    }

    transformations::RPCNodeTransform rpc_transform(loop, target_, category_, rpc_context_);

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
    if (loop_info.loop_info.loopnest_index == -1 || loop_info.loop_info.has_side_effects) {
        return scheduler::NEXT;
    }

    // Apply transfer tuning to the loop
    transformations::RPCNodeTransform rpc_transform(loop, target_, category_, rpc_context_, print_steps_);
    if (rpc_transform.can_be_applied(builder, analysis_manager)) {
        rpc_transform.apply(builder, analysis_manager);
        return scheduler::NEXT;
    }

    return scheduler::NEXT;
}

bool RpcLoopOpt::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    auto& flop_analysis = analysis_manager.get<analysis::FlopAnalysis>();

    // Initialize queue with outermost loops
    std::list<structured_control_flow::ControlFlowNode*> queue;
    std::unordered_map<structured_control_flow::ControlFlowNode*, scheduler::SchedulerLoopInfo> scheduling_info_map;
    for (auto& loop : loop_analysis.outermost_loops()) {
        queue.push_back(loop);

        scheduler::SchedulerLoopInfo info;
        info.loop_info = loop_analysis.loop_info(loop);
        info.flop = flop_analysis.get(loop);
        scheduling_info_map[loop] = info;
    }
    if (queue.empty()) {
        return false;
    }

    // Scheduling state machine
    bool applied = false;
    while (!queue.empty()) {
        auto loop = queue.front();
        queue.pop_front();

        auto scheduling_info = scheduling_info_map.at(loop);
        scheduling_info_map.erase(loop);

        scheduler::SchedulerAction action;
        if (scheduling_info.loop_info.has_side_effects) {
            action = scheduler::SchedulerAction::NEXT;
        } else if (auto while_loop = dynamic_cast<structured_control_flow::While*>(loop)) {
            action = schedule(builder, analysis_manager, *while_loop, scheduling_info);
        } else if (auto structured_loop = dynamic_cast<structured_control_flow::StructuredLoop*>(loop)) {
            action = schedule(builder, analysis_manager, *structured_loop, scheduling_info);
        } else {
            throw InvalidSDFGException("LoopScheduler encountered non-loop in loop analysis.");
        }

        switch (action) {
            case scheduler::SchedulerAction::NEXT: {
                applied = true;
                break;
            }
            // exchanging the sdfg is a non-local operation that requires fresh analysis data
            case scheduler::SchedulerAction::CHILDREN: {
                break;
            }
        }
    }

    return applied;
}

} // namespace rpc
} // namespace passes
} // namespace sdfg
