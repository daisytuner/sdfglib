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
    bool can_be_applied(const structured_control_flow::For& for_stmt,
                        analysis::AnalysisManager& analysis_manager);
};

}  // namespace passes
}  // namespace sdfg
