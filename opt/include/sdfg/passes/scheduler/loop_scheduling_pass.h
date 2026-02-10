#pragma once

#include <sdfg/analysis/flop_analysis.h>
#include <sdfg/analysis/loop_analysis.h>
#include <sdfg/passes/pass.h>
#include <string>
#include "sdfg/optimization_report/pass_report_consumer.h"

namespace sdfg {
namespace passes {
namespace scheduler {

class LoopSchedulingPass : public Pass {
private:
    std::vector<std::string> targets_;
    sdfg::PassReportConsumer* report_;
    bool offload_unknown_sizes_;

    bool run_pass_target(
        builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager, const std::string& target
    );

public:
    LoopSchedulingPass(
        const std::vector<std::string>& targets, sdfg::PassReportConsumer* report, bool offload_unknown_sizes = false
    )
        : targets_(targets), report_(report), offload_unknown_sizes_(offload_unknown_sizes) {}
    ~LoopSchedulingPass() override = default;

    bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    std::string name() override { return "LoopSchedulingPass"; }
};


} // namespace scheduler
} // namespace passes
} // namespace sdfg
