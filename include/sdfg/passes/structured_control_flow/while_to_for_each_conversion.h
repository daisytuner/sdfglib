#pragma once

#include "sdfg/passes/pass.h"

namespace sdfg {
namespace passes {

class WhileToForEachConversion : public Pass {
   private:
    bool can_be_applied(builder::StructuredSDFGBuilder& builder,
                        analysis::AnalysisManager& analysis_manager,
                        structured_control_flow::While& loop);

    void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager,
               structured_control_flow::Sequence& parent, structured_control_flow::While& loop);

   public:
    WhileToForEachConversion();

    std::string name() override;

    virtual bool run_pass(builder::StructuredSDFGBuilder& builder,
                          analysis::AnalysisManager& analysis_manager) override;
};

}  // namespace passes
}  // namespace sdfg
