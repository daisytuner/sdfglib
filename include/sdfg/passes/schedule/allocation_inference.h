#pragma once

#include "sdfg/passes/pass.h"

namespace sdfg {
namespace passes {

class AllocationInference : public Pass {
   private:
   public:
    AllocationInference();

    std::string name() override;

    virtual bool run_pass(Schedule& schedule) override;
};

}  // namespace passes
}  // namespace sdfg
