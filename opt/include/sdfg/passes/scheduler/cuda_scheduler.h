#pragma once

#include "sdfg/passes/scheduler/loop_scheduler.h"

namespace sdfg {
namespace passes {
namespace scheduler {

class CUDAScheduler : public LoopScheduler {
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
    std::string name() override { return "CUDAScheduler"; };
};

} // namespace scheduler
} // namespace passes
} // namespace sdfg
