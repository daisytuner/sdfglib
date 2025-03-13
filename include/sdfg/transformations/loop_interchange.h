#pragma once

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/data_parallelism_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

class LoopInterchange : public Transformation {
    structured_control_flow::For& outer_loop_;
    structured_control_flow::For& inner_loop_;

   public:
    LoopInterchange(structured_control_flow::Sequence& parent,
                    structured_control_flow::For& outer_loop,
                    structured_control_flow::For& inner_loop);

    virtual std::string name() override;

    virtual bool can_be_applied(Schedule& schedule) override;

    virtual void apply(Schedule& schedule) override;
};

}  // namespace transformations
}  // namespace sdfg
