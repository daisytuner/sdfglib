#include "sdfg/passes/scheduler/transfer_tuning_scheduler.h"

#include "sdfg/transformations/local_transfertuning_transform.h"

namespace sdfg {
namespace passes {
namespace scheduler {

SchedulerAction TransferTuningScheduler::schedule(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::StructuredLoop& loop,
    const SchedulerLoopInfo& loop_info
) {
    // Apply transfer tuning to the loop
    transformations::LocalTransferTuningTransform
        transfer_tuning_transform("sequential", "server", &builder.subject(), loop_info.loop_info);

    if (transfer_tuning_transform.can_be_applied(builder, analysis_manager)) {
        transfer_tuning_transform.apply(builder, analysis_manager);
        return NEXT;
    }

    // Check if in not outermost loop
    if (loop_info.loop_info.loopnest_index == -1) {
        return NEXT;
    } else {
        // Visit 1st-level children
        return CHILDREN;
    }
}

SchedulerAction TransferTuningScheduler::schedule(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::While& loop,
    const SchedulerLoopInfo& loop_info
) {
    // Apply transfer tuning to the loop
    transformations::LocalTransferTuningTransform
        transfer_tuning_transform("sequential", "server", &builder.subject(), loop_info.loop_info);
    if (transfer_tuning_transform.can_be_applied(builder, analysis_manager)) {
        transfer_tuning_transform.apply(builder, analysis_manager);
        return NEXT;
    }

    // Check if in not outermost loop
    if (loop_info.loop_info.loopnest_index == -1) {
        return NEXT;
    } else {
        // Visit 1st-level children
        return CHILDREN;
    }
}

} // namespace scheduler
} // namespace passes
} // namespace sdfg
