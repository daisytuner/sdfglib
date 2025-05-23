#pragma once

#include <tuple>

#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

class KernelLocalStorage : public Transformation {
    structured_control_flow::Sequence& parent_;
    structured_control_flow::StructuredLoop& outer_loop_;
    structured_control_flow::StructuredLoop& inner_loop_;
    std::string container_;

   public:
    KernelLocalStorage(structured_control_flow::Sequence& parent,
                       structured_control_flow::StructuredLoop& outer_loop,
                       structured_control_flow::StructuredLoop& inner_loop, std::string container);

    virtual std::string name() override;

    virtual bool can_be_applied(Schedule& schedule) override;

    virtual void apply(Schedule& schedule) override;

   private:
    bool reads_container(std::string container, const structured_control_flow::Sequence& sequence,
                         analysis::UsersView& body_users);
    bool uses_inner_indvar(const structured_control_flow::Kernel* kernel,
                           const structured_control_flow::Sequence& body,
                           analysis::UsersView& body_users);
    std::tuple<symbolic::Integer, symbolic::Integer, symbolic::Integer> dim_size(
        const structured_control_flow::Kernel* kernel, symbolic::Assumptions& assumptions);
};

}  // namespace transformations
}  // namespace sdfg
