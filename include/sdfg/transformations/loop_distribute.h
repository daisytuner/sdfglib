#pragma once

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/data_parallelism_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/deepcopy/structured_sdfg_deep_copy.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

class LoopDistribute : public Transformation {
    structured_control_flow::Sequence& parent_;
    structured_control_flow::For& loop_;

   public:
    LoopDistribute(structured_control_flow::Sequence& parent, structured_control_flow::For& loop);

    virtual std::string name() override;

    virtual bool can_be_applied(Schedule& schedule) override;

    virtual void apply(Schedule& schedule) override;
};

}  // namespace transformations
}  // namespace sdfg
