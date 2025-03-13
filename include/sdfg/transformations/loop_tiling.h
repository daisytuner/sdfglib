#pragma once

#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

class LoopTiling : public Transformation {
    structured_control_flow::Sequence& parent_;
    structured_control_flow::For& loop_;
    size_t tile_size_;

   public:
    LoopTiling(structured_control_flow::Sequence& parent, structured_control_flow::For& loop,
               size_t tile_size);

    virtual std::string name() override;

    virtual bool can_be_applied(Schedule& schedule) override;

    virtual void apply(Schedule& schedule) override;
};

}  // namespace transformations
}  // namespace sdfg
