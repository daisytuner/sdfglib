#pragma once

#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

class OutLocalStorage : public Transformation {
    structured_control_flow::Sequence& parent_;
    structured_control_flow::For& loop_;
    std::string container_;
    bool requires_array_;

   public:
    OutLocalStorage(structured_control_flow::Sequence& parent, structured_control_flow::For& loop,
                    std::string container);

    virtual std::string name() override;

    virtual bool can_be_applied(Schedule& schedule) override;

    virtual void apply(Schedule& schedule) override;

   private:
    void apply_array(Schedule& schedule);

    void apply_scalar(Schedule& schedule);
};

}  // namespace transformations
}  // namespace sdfg
