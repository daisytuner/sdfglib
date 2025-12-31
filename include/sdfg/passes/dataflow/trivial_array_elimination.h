#pragma once

#include "sdfg/passes/pass.h"

namespace sdfg {
namespace passes {

class TrivialArrayElimination : public Pass {
public:
    TrivialArrayElimination();

    virtual std::string name() override;

    virtual bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
};

} // namespace passes
} // namespace sdfg
