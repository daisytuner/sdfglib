#pragma once

#include "sdfg/codegen/utils.h"
#include "sdfg/structured_sdfg.h"

namespace sdfg {
namespace codegen {

class Instrumentation {
   protected:
    StructuredSDFG& sdfg_;

   public:
    Instrumentation(StructuredSDFG& sdfg) : sdfg_(sdfg) {}
    virtual ~Instrumentation() = default;

    virtual bool should_instrument(const structured_control_flow::ControlFlowNode& node) const {
        return false;
    };

    virtual void begin_instrumentation(const structured_control_flow::ControlFlowNode& node,
                                       PrettyPrinter& stream) const {
        // Do nothing by default
    };

    virtual void end_instrumentation(const structured_control_flow::ControlFlowNode& node,
                                     PrettyPrinter& stream) const {
        // Do nothing by default
    };
};

}  // namespace codegen
}  // namespace sdfg