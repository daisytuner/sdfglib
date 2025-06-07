#include "sdfg/schedule.h"

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/deepcopy/structured_sdfg_deep_copy.h"

namespace sdfg {

Schedule::Schedule(std::unique_ptr<StructuredSDFG>& sdfg)
    : assumptions_(), builder_(sdfg), analysis_manager_(builder_.subject(), assumptions_) {

      };

Schedule::Schedule(std::unique_ptr<StructuredSDFG>& sdfg, const symbolic::Assumptions& assumptions)
    : assumptions_(assumptions),
      builder_(sdfg),
      analysis_manager_(builder_.subject(), assumptions_) {

      };

symbolic::Assumptions& Schedule::assumptions() { return this->assumptions_; };

const symbolic::Assumptions& Schedule::assumptions() const { return this->assumptions_; };

builder::StructuredSDFGBuilder& Schedule::builder() { return this->builder_; };

StructuredSDFG& Schedule::sdfg() { return this->builder_.subject(); };

const StructuredSDFG& Schedule::sdfg() const { return this->builder_.subject(); };

analysis::AnalysisManager& Schedule::analysis_manager() { return this->analysis_manager_; };

LoopSchedule Schedule::loop_schedule(const structured_control_flow::ControlFlowNode* loop) const {
    if (this->loop_schedules_.find(loop) == this->loop_schedules_.end()) {
        return LoopSchedule::SEQUENTIAL;
    }
    return this->loop_schedules_.at(loop);
};

void Schedule::loop_schedule(const structured_control_flow::ControlFlowNode* loop,
                             const LoopSchedule schedule) {
    if (schedule == LoopSchedule::SEQUENTIAL) {
        if (this->loop_schedules_.find(loop) == this->loop_schedules_.end()) {
            return;
        }
        this->loop_schedules_.erase(loop);
        return;
    }
    this->loop_schedules_.insert_or_assign(loop, schedule);
};

}  // namespace sdfg
