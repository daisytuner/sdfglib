#pragma once

#include "sdfg/passes/scheduler/loop_scheduler.h"
#include "sdfg/targets/omp/schedule.h"

namespace sdfg {
namespace passes {
namespace scheduler {

class OMPScheduler : public LoopScheduler {
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
    std::string name() override { return "OMPScheduler"; };
};

} // namespace scheduler
} // namespace passes
} // namespace sdfg
