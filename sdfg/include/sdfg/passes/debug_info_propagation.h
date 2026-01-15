#pragma once

#include "sdfg/passes/pass.h"

namespace sdfg {
namespace passes {

class DebugInfoPropagation : public Pass {
private:
    void propagate(structured_control_flow::ControlFlowNode* current);

public:
    DebugInfoPropagation();

    std::string name() override;

    virtual bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
};

} // namespace passes
} // namespace sdfg
