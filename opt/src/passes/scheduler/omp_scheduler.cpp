#include "sdfg/passes/scheduler/omp_scheduler.h"

#include "sdfg/transformations/omp_transform.h"

namespace sdfg {
namespace passes {
namespace scheduler {

SchedulerAction OMPScheduler::schedule(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::StructuredLoop& loop,
    const SchedulerLoopInfo& loop_info
) {
    if (auto map_node = dynamic_cast<structured_control_flow::Map*>(&loop)) {
        // Apply OpenMP parallelization to the loop
        transformations::OMPTransform omp_transform(*map_node);
        if (omp_transform.can_be_applied(builder, analysis_manager)) {
            omp_transform.apply(builder, analysis_manager);
            return NEXT;
        }
    }

    // Check if in not outermost loop
    if (loop_info.loop_info.loopnest_index == -1) {
        return NEXT;
    } else {
        // Visit 1st-level children
        return CHILDREN;
    }
}

SchedulerAction OMPScheduler::schedule(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::While& loop,
    const SchedulerLoopInfo& loop_info
) {
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
