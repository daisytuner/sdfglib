#pragma once

#include "sdfg/passes/rpc/rpc_context.h"
#include "sdfg/passes/scheduler/loop_scheduler.h"

namespace sdfg {
namespace passes {
namespace scheduler {

class RpcLoopOpt : public LoopScheduler {
private:
    rpc::RpcContext& rpc_context_;


protected:
    SchedulerAction schedule(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::StructuredLoop& loop,
        const SchedulerLoopInfo& loop_info
    ) override;

    SchedulerAction schedule(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::While& loop,
        const SchedulerLoopInfo& loop_info
    ) override;

public:
    RpcLoopOpt(rpc::RpcContext& rpc_context);

    std::string name() override { return "RpcLoopOpt"; };
};

} // namespace scheduler
} // namespace passes
} // namespace sdfg
