#pragma once

#include <algorithm>

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/data_parallelism_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/passes/pass.h"
#include "sdfg/transformations/loop_distribute.h"

namespace sdfg {
namespace passes {

class PerfectLoopDistribution : public Pass {
   private:
    bool can_be_applied(Schedule& schedule, structured_control_flow::Sequence& parent,
                        structured_control_flow::For& loop);

    void apply(Schedule& schedule, structured_control_flow::Sequence& parent,
               structured_control_flow::For& loop);

   public:
    PerfectLoopDistribution();

    std::string name() override;

    virtual bool run_pass(Schedule& schedule) override;
};

}  // namespace passes
}  // namespace sdfg
