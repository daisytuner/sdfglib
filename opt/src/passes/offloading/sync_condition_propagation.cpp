#include "sdfg/passes/offloading/sync_condition_propagation.h"

#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/transformations/offloading/gpu_condition_propagation.h"
namespace sdfg::passes {

bool SyncConditionPropagation::
    run_pass(sdfg::builder::StructuredSDFGBuilder& builder, sdfg::analysis::AnalysisManager& analysis_manager) {
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    bool modified = false;

    for (auto loop : loop_analysis.loops()) {
        if (auto map = dynamic_cast<structured_control_flow::Map*>(loop)) {
            if (map->schedule_type().value() == sdfg::cuda::ScheduleType_CUDA::value()) {
                sdfg::transformations::GPUConditionPropagation gpu_condition_propagation(*map);
                if (gpu_condition_propagation.can_be_applied(builder, analysis_manager)) {
                    gpu_condition_propagation.apply(builder, analysis_manager);
                    modified = true;
                }
            }
        }
    }

    return modified;
}

SyncConditionPropagation::SyncConditionPropagation() {}

} // namespace sdfg::passes
