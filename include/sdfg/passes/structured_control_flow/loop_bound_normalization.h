#pragma once

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/passes/pass.h"

namespace sdfg {
namespace passes {

class LoopBoundNormalization : public Pass {
   private:
    bool apply(builder::StructuredSDFGBuilder& builder, structured_control_flow::For& loop);

   public:
    LoopBoundNormalization();

    std::string name() override;

    virtual bool run_pass(builder::StructuredSDFGBuilder& builder,
                          analysis::AnalysisManager& analysis_manager) override;
};

}  // namespace passes
}  // namespace sdfg
