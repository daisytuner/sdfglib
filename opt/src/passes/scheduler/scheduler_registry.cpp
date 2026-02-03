#include "sdfg/passes/scheduler/cuda_scheduler.h"
#include "sdfg/passes/scheduler/highway_scheduler.h"
#include "sdfg/passes/scheduler/omp_scheduler.h"
#include "sdfg/passes/scheduler/scheduler_registry.h"

namespace sdfg {
namespace passes {
namespace scheduler {

std::unique_ptr<LoopScheduler> create_loop_scheduler(
    const std::string target
) {
    auto scheduler = SchedulerRegistry::instance().get_loop_scheduler(target);
    if (scheduler) {
        return std::unique_ptr<LoopScheduler>(scheduler);
    }
    
    throw std::runtime_error("Unsupported scheduling target: " + target);
};

void register_default_schedulers() {
    SchedulerRegistry::instance().register_loop_scheduler<CUDAScheduler>(CUDAScheduler::target());
    SchedulerRegistry::instance().register_loop_scheduler<OMPScheduler>(OMPScheduler::target());
    SchedulerRegistry::instance().register_loop_scheduler<HighwayScheduler>(HighwayScheduler::target());
}

} // namespace scheduler
} // namespace passes
} // namespace sdfg