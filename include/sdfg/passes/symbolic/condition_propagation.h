#pragma once

#include "sdfg/passes/pass.h"

namespace sdfg {
namespace passes {

class ForwardConditionPropagation : public Pass {
   private:
    bool propagate_condition(builder::StructuredSDFGBuilder& builder,
                             analysis::AnalysisManager& analysis_manager,
                             structured_control_flow::Sequence& parent,
                             const symbolic::Condition& condition);

   public:
    ForwardConditionPropagation();

    virtual std::string name() override;

    virtual bool run_pass(builder::StructuredSDFGBuilder& builder,
                          analysis::AnalysisManager& analysis_manager) override;
};

class BackwardConditionPropagation : public Pass {
   private:
    bool eliminate_condition(builder::StructuredSDFGBuilder& builder,
                             structured_control_flow::Sequence& root,
                             structured_control_flow::IfElse& match,
                             structured_control_flow::For& loop,
                             const symbolic::Condition& condition);

   public:
    BackwardConditionPropagation();

    std::string name() override;

    virtual bool run_pass(builder::StructuredSDFGBuilder& builder,
                          analysis::AnalysisManager& analysis_manager) override;
};

}  // namespace passes
}  // namespace sdfg
