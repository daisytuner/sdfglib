#pragma once

#include "sdfg/passes/rpc/rpc_context.h"
#include "sdfg/passes/scheduler/loop_scheduler.h"

namespace sdfg {
namespace passes {
namespace rpc {

class RpcLoopOpt : public scheduler::LoopScheduler {
private:
    rpc::RpcContext& rpc_context_;
    const std::string target_;
    const std::string category_;

protected:
    scheduler::SchedulerAction schedule(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::StructuredLoop& loop,
        const scheduler::SchedulerLoopInfo& loop_info
    ) override;

    scheduler::SchedulerAction schedule(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::While& loop,
        const scheduler::SchedulerLoopInfo& loop_info
    ) override;

public:
    RpcLoopOpt(rpc::RpcContext& rpc_context, std::string target, std::string category);

    std::string name() override { return "RpcLoopOpt"; };
};

} // namespace rpc
} // namespace passes
} // namespace sdfg
