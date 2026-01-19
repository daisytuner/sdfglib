#include "sdfg/passes/scheduler/cuda_scheduler.h"

#include "sdfg/transformations/offloading/cuda_parallelize_nested_map.h"
#include "sdfg/transformations/offloading/cuda_transform.h"

namespace sdfg {
namespace passes {
namespace scheduler {

SchedulerAction CUDAScheduler::schedule(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::StructuredLoop& loop,
    const SchedulerLoopInfo& loop_info
) {
    if (auto map_node = dynamic_cast<structured_control_flow::Map*>(&loop)) {
        // Apply OpenMP parallelization to the loop
        cuda::CUDATransform cuda_transform(*map_node, 32, false);
        auto cuda_plan = cuda_transform.can_be_applied(builder, analysis_manager);
        if (cuda_plan) {
            cuda_transform.apply(builder, analysis_manager);

            auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
            auto descendants = loop_analysis.descendants(map_node);
            for (auto& descendant : descendants) {
                if (auto nested_map = dynamic_cast<structured_control_flow::Map*>(descendant)) {
                    transformations::CUDAParallelizeNestedMap nested_cuda_transform(*nested_map, 8);
                    if (nested_cuda_transform.can_be_applied(builder, analysis_manager)) {
                        nested_cuda_transform.apply(builder, analysis_manager);
                    }
                }
            }

            analysis_manager.invalidate_all();
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

SchedulerAction CUDAScheduler::schedule(
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
