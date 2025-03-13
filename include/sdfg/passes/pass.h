#pragma once

#include <sdfg/analysis/analysis.h>

#include "sdfg/builder/sdfg_builder.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/conditional_schedule.h"
#include "sdfg/schedule.h"

namespace sdfg {
namespace passes {

class Pass {
   public:
    virtual std::string name() = 0;

    bool run(builder::SDFGBuilder& builder);

    bool run(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    bool run(Schedule& schedule);

    bool run(ConditionalSchedule& schedule);

    virtual bool run_pass(builder::SDFGBuilder& builder);

    virtual bool run_pass(builder::StructuredSDFGBuilder& builder,
                          analysis::AnalysisManager& analysis_manager);

    virtual bool run_pass(Schedule& schedule);

    virtual void invalidates(analysis::AnalysisManager& analysis_manager, bool applied);
};

template <typename T>
class VisitorPass : public Pass {
    std::string name() override { return "VisitorPass"; };

    bool run_pass(builder::StructuredSDFGBuilder& builder,
                  analysis::AnalysisManager& analysis_manager) override {
        T visitor(builder, analysis_manager);
        return visitor.visit();
    };
};

}  // namespace passes
}  // namespace sdfg
