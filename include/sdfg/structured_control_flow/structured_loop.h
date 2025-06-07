#pragma once

#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {

namespace builder {
class StructuredSDFGBuilder;
}

namespace structured_control_flow {

class StructuredLoop : public ControlFlowNode {
    friend class sdfg::builder::StructuredSDFGBuilder;

   protected:
    StructuredLoop(const DebugInfo& debug_info);

   public:
    virtual ~StructuredLoop() = default;

    StructuredLoop(const StructuredLoop& node) = delete;
    StructuredLoop& operator=(const StructuredLoop&) = delete;

    virtual const symbolic::Symbol& indvar() const = 0;

    virtual const symbolic::Expression& init() const = 0;

    virtual const symbolic::Expression& update() const = 0;

    virtual const symbolic::Condition& condition() const = 0;

    virtual Sequence& root() const = 0;
};

}  // namespace structured_control_flow
}  // namespace sdfg