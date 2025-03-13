#pragma once

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/data_parallelism_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/passes/pass.h"

namespace sdfg {
namespace passes {

class LoopDependentSymbolElimination : public Pass {
   private:
    bool eliminate_symbols(builder::StructuredSDFGBuilder& builder,
                           analysis::AnalysisManager& analysis_manager,
                           structured_control_flow::For& loop,
                           structured_control_flow::Transition& transition);

   public:
    LoopDependentSymbolElimination();

    std::string name() override;

    virtual bool run_pass(builder::StructuredSDFGBuilder& builder,
                          analysis::AnalysisManager& analysis_manager) override;
};

}  // namespace passes
}  // namespace sdfg
