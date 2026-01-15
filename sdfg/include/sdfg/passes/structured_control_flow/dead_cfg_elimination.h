#pragma once

#include "sdfg/passes/pass.h"

namespace sdfg {
namespace passes {

class DeadCFGElimination : public Pass {
private:
    bool is_dead(const structured_control_flow::ControlFlowNode& node);

public:
    DeadCFGElimination();

    virtual std::string name() override;

    virtual bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
};

} // namespace passes
} // namespace sdfg
