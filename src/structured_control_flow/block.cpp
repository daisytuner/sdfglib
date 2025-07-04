#include "sdfg/structured_control_flow/block.h"

#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace structured_control_flow {

Block::Block(size_t element_id, const DebugInfo& debug_info) : ControlFlowNode(element_id, debug_info) {
    this->dataflow_ = std::make_unique<data_flow::DataFlowGraph>();
};

const data_flow::DataFlowGraph& Block::dataflow() const { return *this->dataflow_; };

data_flow::DataFlowGraph& Block::dataflow() { return *this->dataflow_; };

void Block::replace(const symbolic::Expression& old_expression, const symbolic::Expression& new_expression) {
    this->dataflow_->replace(old_expression, new_expression);
};

} // namespace structured_control_flow
} // namespace sdfg
