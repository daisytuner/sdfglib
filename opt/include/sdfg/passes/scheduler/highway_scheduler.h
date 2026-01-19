#pragma once

#include "sdfg/passes/scheduler/loop_scheduler.h"
#include "sdfg/targets/highway/schedule.h"

namespace sdfg {
namespace passes {
namespace scheduler {

class HighwayScheduler : public LoopScheduler {
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
    std::string name() override { return "HighwayScheduler"; };

    bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
};

} // namespace scheduler
} // namespace passes
} // namespace sdfg
