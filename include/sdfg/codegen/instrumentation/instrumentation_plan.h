#pragma once

#include "sdfg/codegen/utils.h"
#include "sdfg/structured_sdfg.h"

namespace sdfg {
namespace codegen {

enum InstrumentationEventType {
    CPU = 0,
    CUDA = 1,
};

class InstrumentationPlan {
protected:
    StructuredSDFG& sdfg_;
    std::unordered_map<const structured_control_flow::ControlFlowNode*, InstrumentationEventType> nodes_;

public:
    InstrumentationPlan(
        StructuredSDFG& sdfg,
        const std::unordered_map<const structured_control_flow::ControlFlowNode*, InstrumentationEventType>& nodes
    )
        : sdfg_(sdfg), nodes_(nodes) {}

    InstrumentationPlan(const InstrumentationPlan& other) = delete;
    InstrumentationPlan(InstrumentationPlan&& other) = delete;

    InstrumentationPlan& operator=(const InstrumentationPlan& other) = delete;
    InstrumentationPlan& operator=(InstrumentationPlan&& other) = delete;

    void update(const structured_control_flow::ControlFlowNode& node, InstrumentationEventType event_type);

    bool is_empty() const { return nodes_.empty(); }

    bool should_instrument(const structured_control_flow::ControlFlowNode& node) const;

    void begin_instrumentation(const structured_control_flow::ControlFlowNode& node, PrettyPrinter& stream) const;

    void end_instrumentation(const structured_control_flow::ControlFlowNode& node, PrettyPrinter& stream) const;

    static std::unique_ptr<InstrumentationPlan> none(StructuredSDFG& sdfg);

    static std::unique_ptr<InstrumentationPlan> outermost_loops_plan(StructuredSDFG& sdfg);
};

} // namespace codegen
} // namespace sdfg
