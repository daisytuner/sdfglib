#pragma once

#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

enum AccessType { CONSTANT, CONTIGUOUS, INDIRECTION, UNKNOWN };

class Vectorization : public Transformation {
    structured_control_flow::StructuredLoop& loop_;

    static bool tasklet_supported(data_flow::TaskletCode c);

   public:
    Vectorization(structured_control_flow::Sequence& parent,
                  structured_control_flow::StructuredLoop& loop);

    virtual std::string name() override;

    virtual bool can_be_applied(Schedule& schedule) override;

    virtual void apply(Schedule& schedule) override;

    static AccessType classify_access(const data_flow::Subset& subset,
                                      const symbolic::Symbol& indvar,
                                      const std::unordered_set<std::string>& moving_symbols);
};

}  // namespace transformations
}  // namespace sdfg
