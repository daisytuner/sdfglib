#pragma once

#include "sdfg/passes/pass.h"
#include "sdfg/structured_control_flow/sequence.h"

namespace sdfg {
namespace passes {

class For2Map : public Pass {
   public:
    For2Map();

    virtual std::string name() override;

    virtual bool run_pass(builder::StructuredSDFGBuilder& builder,
                          analysis::AnalysisManager& analysis_manager) override;

   private:
    symbolic::Expression num_iterations(const structured_control_flow::For& for_stmt,
                                        analysis::AnalysisManager& analysis_manager) const;
};

}  // namespace passes
}  // namespace sdfg
