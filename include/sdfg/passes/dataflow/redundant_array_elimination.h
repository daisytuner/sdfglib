#pragma once

#include "sdfg/passes/pass.h"

namespace sdfg {
namespace passes {

class RedundantArrayElimination : public Pass {
public:
    RedundantArrayElimination();

    virtual std::string name() override;

    virtual bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
};

} // namespace passes
} // namespace sdfg
