#include "sdfg/control_flow/state.h"

namespace sdfg {
namespace control_flow {

State::State(size_t element_id, const DebugInfo& debug_info, const graph::Vertex vertex)
    : Element(element_id, debug_info), vertex_(vertex) {
    this->dataflow_ = std::make_unique<data_flow::DataFlowGraph>();
};

graph::Vertex State::vertex() const { return this->vertex_; };

const data_flow::DataFlowGraph& State::dataflow() const { return *this->dataflow_; };

data_flow::DataFlowGraph& State::dataflow() { return *this->dataflow_; };

void State::replace(const symbolic::Expression& old_expression,
                    const symbolic::Expression& new_expression) {
    this->dataflow_->replace(old_expression, new_expression);
};

}  // namespace control_flow
}  // namespace sdfg
