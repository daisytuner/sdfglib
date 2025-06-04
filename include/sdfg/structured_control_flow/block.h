#pragma once

#include <memory>

#include "sdfg/control_flow/state.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {

namespace builder {
class StructuredSDFGBuilder;
}

namespace structured_control_flow {

class Block : public ControlFlowNode {
    friend class sdfg::builder::StructuredSDFGBuilder;

   private:
    std::unique_ptr<data_flow::DataFlowGraph> dataflow_;

    Block(const DebugInfo& debug_info);

    Block(const DebugInfo& debug_info, const data_flow::DataFlowGraph& dataflow);

   public:
    Block(const Block& block) = delete;
    Block& operator=(const Block&) = delete;

    const data_flow::DataFlowGraph& dataflow() const;

    data_flow::DataFlowGraph& dataflow();

    void replace(const symbolic::Expression& old_expression,
                 const symbolic::Expression& new_expression) override;
};

}  // namespace structured_control_flow
}  // namespace sdfg
