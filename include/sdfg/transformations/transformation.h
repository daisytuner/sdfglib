#pragma once

#include "sdfg/analysis/analysis.h"
#include "sdfg/schedule.h"

namespace sdfg {
namespace transformations {

class Transformation {
   public:
    virtual ~Transformation() = default;

    virtual std::string name() = 0;

    virtual bool can_be_applied(Schedule& schedule);

    virtual void apply(Schedule& schedule);

    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder,
                                analysis::AnalysisManager& analysis_manager);

    virtual void apply(builder::StructuredSDFGBuilder& builder,
                       analysis::AnalysisManager& analysis_manager);
};

}  // namespace transformations
}  // namespace sdfg
