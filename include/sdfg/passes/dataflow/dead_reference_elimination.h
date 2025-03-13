#pragma once

#include "sdfg/analysis/data_parallelism_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/passes/pass.h"

namespace sdfg {
namespace passes {

class DeadReferenceElimination : public Pass {
   public:
    DeadReferenceElimination();

    virtual std::string name() override;

    virtual bool run_pass(builder::StructuredSDFGBuilder& builder,
                          analysis::AnalysisManager& analysis_manager) override;
};

}  // namespace passes
}  // namespace sdfg
