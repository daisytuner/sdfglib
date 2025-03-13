#pragma once

#include "sdfg/structured_control_flow/control_flow_node.h"

namespace sdfg {

namespace builder {
class StructuredSDFGBuilder;
}

namespace structured_control_flow {

class Return : public ControlFlowNode {
    friend class sdfg::builder::StructuredSDFGBuilder;

    Return(size_t element_id, const DebugInfo& debug_info);

   public:
    Return(const Return& Return) = delete;
    Return& operator=(const Return&) = delete;

    void replace(const symbolic::Expression& old_expression,
                 const symbolic::Expression& new_expression) override;
};

}  // namespace structured_control_flow
}  // namespace sdfg
