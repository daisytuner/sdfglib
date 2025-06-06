#include "sdfg/structured_control_flow/block.h"

#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace structured_control_flow {

Block::Block(const DebugInfo& debug_info) : ControlFlowNode(debug_info) {
    this->dataflow_ = std::make_unique<data_flow::DataFlowGraph>();
};

Block::Block(const DebugInfo& debug_info, const data_flow::DataFlowGraph& dataflow)
    : ControlFlowNode(debug_info) {
    this->dataflow_ = dataflow.clone();
};

const data_flow::DataFlowGraph& Block::dataflow() const { return *this->dataflow_; };

data_flow::DataFlowGraph& Block::dataflow() { return *this->dataflow_; };

void Block::replace(const symbolic::Expression& old_expression,
                    const symbolic::Expression& new_expression) {
    this->dataflow_->replace(old_expression, new_expression);
};

}  // namespace structured_control_flow
}  // namespace sdfg
