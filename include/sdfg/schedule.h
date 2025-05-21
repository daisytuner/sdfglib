#pragma once

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_sdfg.h"

namespace sdfg {

enum LoopSchedule { SEQUENTIAL, VECTORIZATION, MULTICORE };

enum AllocationType {
    DECLARE,
    ALLOCATE,
};

class Schedule {
   private:
    symbolic::Assumptions assumptions_;

    builder::StructuredSDFGBuilder builder_;
    analysis::AnalysisManager analysis_manager_;

    std::unordered_map<const structured_control_flow::ControlFlowNode*, LoopSchedule>
        loop_schedules_;

    std::unordered_map<std::string, AllocationType> allocation_types_;
    std::unordered_map<std::string, const structured_control_flow::ControlFlowNode*>
        allocation_lifetimes_;

   public:
    Schedule(std::unique_ptr<StructuredSDFG>& sdfg);
    Schedule(std::unique_ptr<StructuredSDFG>& sdfg, const symbolic::Assumptions& assumptions);

    Schedule(const Schedule& schedule) = delete;
    Schedule& operator=(const Schedule&) = delete;

    symbolic::Assumptions& assumptions();

    const symbolic::Assumptions& assumptions() const;

    builder::StructuredSDFGBuilder& builder();

    const StructuredSDFG& sdfg() const;

    analysis::AnalysisManager& analysis_manager();

    LoopSchedule loop_schedule(const structured_control_flow::ControlFlowNode* loop) const;

    void loop_schedule(const structured_control_flow::ControlFlowNode* loop,
                       const LoopSchedule schedule);

    /***** Allocation Management *****/

    const structured_control_flow::ControlFlowNode* allocation_lifetime(
        const std::string& container) const;

    void allocation_lifetime(const std::string& container,
                             const structured_control_flow::ControlFlowNode* node);

    AllocationType allocation_type(const std::string& container) const;

    void allocation_type(const std::string& container, AllocationType allocation_type);

    std::unordered_set<std::string> node_allocations(
        const structured_control_flow::ControlFlowNode* node) const;

    std::unordered_set<std::string> allocations(
        const structured_control_flow::ControlFlowNode* node) const;
};
}  // namespace sdfg
