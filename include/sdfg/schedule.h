#pragma once

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_sdfg.h"

namespace sdfg {

enum LoopSchedule { SEQUENTIAL, VECTORIZATION, MULTICORE };

class Schedule {
   private:
    symbolic::Assumptions assumptions_;

    builder::StructuredSDFGBuilder builder_;
    analysis::AnalysisManager analysis_manager_;

    std::unordered_map<const structured_control_flow::ControlFlowNode*, LoopSchedule>
        loop_schedules_;

   public:
    Schedule(std::unique_ptr<StructuredSDFG>& sdfg);
    Schedule(std::unique_ptr<StructuredSDFG>& sdfg, const symbolic::Assumptions& assumptions);

    Schedule(const Schedule& schedule) = delete;
    Schedule& operator=(const Schedule&) = delete;

    symbolic::Assumptions& assumptions();

    const symbolic::Assumptions& assumptions() const;

    builder::StructuredSDFGBuilder& builder();

    StructuredSDFG& sdfg();

    const StructuredSDFG& sdfg() const;

    analysis::AnalysisManager& analysis_manager();

    LoopSchedule loop_schedule(const structured_control_flow::ControlFlowNode* loop) const;

    void loop_schedule(const structured_control_flow::ControlFlowNode* loop,
                       const LoopSchedule schedule);
};
}  // namespace sdfg
