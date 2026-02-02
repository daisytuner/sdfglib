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
    const bool print_steps_;

protected:
    scheduler::SchedulerAction schedule(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::StructuredLoop& loop
    ) override;

    scheduler::SchedulerAction schedule(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::While& loop
    ) override;

public:
    RpcLoopOpt(rpc::RpcContext& rpc_context, std::string target, std::string category, bool print_steps = false);

    std::string name() override { return "RpcLoopOpt"; };

    bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

};

} // namespace rpc
} // namespace passes
} // namespace sdfg
