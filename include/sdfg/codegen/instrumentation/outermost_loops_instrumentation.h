#pragma once

#include "sdfg/codegen/instrumentation/instrumentation.h"

namespace sdfg {
namespace codegen {

class OutermostLoopsInstrumentation : public Instrumentation {
    private:
    std::unordered_set<const structured_control_flow::ControlFlowNode*> outermost_loops_;

    public:
    OutermostLoopsInstrumentation(Schedule& schedule);

    bool should_instrument(const structured_control_flow::ControlFlowNode& node) const override;

    void begin_instrumentation(const structured_control_flow::ControlFlowNode& node, PrettyPrinter& stream) const override;

    void end_instrumentation(const structured_control_flow::ControlFlowNode& node, PrettyPrinter& stream) const override;

};

}
}