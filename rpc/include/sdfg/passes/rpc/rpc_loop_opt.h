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

    static std::string target() { return "rpc"; }
};

void register_rpc_loop_opt(rpc::RpcContext& rpc_context, const std::string& target, const std::string& category, bool print_steps = false);

} // namespace rpc
} // namespace passes
} // namespace sdfg
