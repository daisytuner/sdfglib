#pragma once

#include "sdfg/passes/pass.h"

namespace sdfg {
namespace passes {

class RemoveRedundantTransfersPass : public Pass {
public:
    RemoveRedundantTransfersPass();

    std::string name() override;

    bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
};

} // namespace passes
} // namespace sdfg
