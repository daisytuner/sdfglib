#pragma once

#include "sdfg/schedule.h"
#include "sdfg/codegen/utils.h"

namespace sdfg {
namespace codegen {

class Instrumentation {
    protected:
    Schedule& schedule_;

    public:
    Instrumentation(Schedule& schedule) : schedule_(schedule) {}
    virtual ~Instrumentation() = default;

    virtual bool should_instrument(const structured_control_flow::ControlFlowNode& node) const {
        return false;
    };

    virtual void begin_instrumentation(const structured_control_flow::ControlFlowNode& node, PrettyPrinter& stream) const {
        // Do nothing by default
    };

    virtual void end_instrumentation(const structured_control_flow::ControlFlowNode& node, PrettyPrinter& stream) const {
        // Do nothing by default
    };
};

}
}