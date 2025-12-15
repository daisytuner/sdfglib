#pragma once

#include "sdfg/analysis/data_dependency_analysis.h"
#include "sdfg/passes/pass.h"

namespace sdfg {
namespace passes {

class For2MapPass : public Pass {
private:
    std::unique_ptr<analysis::DataDependencyAnalysis> data_dependency_analysis_;

    bool can_be_applied(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::For& for_stmt
    );

public:
    virtual std::string name() override;

    virtual bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
};

} // namespace passes
} // namespace sdfg
