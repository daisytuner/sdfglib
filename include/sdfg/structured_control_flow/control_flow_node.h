#pragma once

#include <boost/lexical_cast.hpp>
#include <memory>
#include <string>

#include "sdfg/element.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {

namespace builder {
class StructuredSDFGBuilder;
}

namespace structured_control_flow {

class ControlFlowNode : public Element {
    friend class builder::StructuredSDFGBuilder;

   protected:
    ControlFlowNode(const DebugInfo& debug_info);

   public:
    virtual ~ControlFlowNode() = default;

    ControlFlowNode(const ControlFlowNode& node) = delete;
    ControlFlowNode& operator=(const ControlFlowNode&) = delete;
};

}  // namespace structured_control_flow
}  // namespace sdfg
