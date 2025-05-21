#pragma once

#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

class For2Map : public Transformation {
    structured_control_flow::Sequence& parent_;
    structured_control_flow::For& loop_;

   public:
    For2Map(structured_control_flow::Sequence& parent, structured_control_flow::For& loop);

    virtual std::string name() override;

    virtual bool can_be_applied(Schedule& schedule) override;

    virtual void apply(Schedule& schedule) override;
};

}  // namespace transformations
}  // namespace sdfg
