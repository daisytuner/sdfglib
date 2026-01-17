#pragma once

#include <sdfg/analysis/flop_analysis.h>
#include <sdfg/analysis/loop_analysis.h>
#include <sdfg/passes/pass.h>

namespace sdfg {
namespace passes {
namespace scheduler {

enum SchedulerAction {
    NEXT,
    CHILDREN,
};

struct SchedulerLoopInfo {
    // Static Properties
    analysis::LoopInfo loop_info = analysis::LoopInfo();

    // Static Analysis
    symbolic::Expression flop = SymEngine::null;
};

class LoopScheduler : public Pass {
protected:
    virtual SchedulerAction schedule(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::StructuredLoop& loop,
        const SchedulerLoopInfo& loop_info
    ) = 0;

    virtual SchedulerAction schedule(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::While& loop,
        const SchedulerLoopInfo& loop_info
    ) = 0;

public:
    ~LoopScheduler() override = default;

    bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
};

} // namespace scheduler
} // namespace passes
} // namespace sdfg
