#pragma once

#include "sdfg/analysis/users.h"
#include "sdfg/passes/pass.h"
#include "sdfg/types/utils.h"

namespace sdfg {
namespace passes {

class ForwardMemletPropagation : public Pass {
   public:
    ForwardMemletPropagation();

    std::string name() override;

    virtual bool run_pass(builder::StructuredSDFGBuilder& builder,
                          analysis::AnalysisManager& analysis_manager) override;
};

class BackwardMemletPropagation : public Pass {
   public:
    BackwardMemletPropagation();

    std::string name() override;

    virtual bool run_pass(builder::StructuredSDFGBuilder& builder,
                          analysis::AnalysisManager& analysis_manager) override;
};

}  // namespace passes
}  // namespace sdfg
