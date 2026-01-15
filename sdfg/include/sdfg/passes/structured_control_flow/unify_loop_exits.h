#pragma once

#include "sdfg/passes/pass.h"

namespace sdfg {
namespace passes {

class UnifyLoopExits : public Pass {
    std::unordered_set<const control_flow::State*>
    determine_loop_nodes(const SDFG& sdfg, const control_flow::State& start, const control_flow::State& end) const;

public:
    virtual std::string name() override;

    virtual bool run_pass(builder::SDFGBuilder& builder) override;
};

} // namespace passes
} // namespace sdfg
