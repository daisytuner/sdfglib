#pragma once

#include "sdfg/passes/pass.h"

namespace sdfg {
namespace passes {

class LoopNormalization : public Pass {
   private:
    bool apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager,
               structured_control_flow::For& loop);

   public:
    LoopNormalization();

    std::string name() override;

    virtual bool run_pass(builder::StructuredSDFGBuilder& builder,
                          analysis::AnalysisManager& analysis_manager) override;
};

}  // namespace passes
}  // namespace sdfg
