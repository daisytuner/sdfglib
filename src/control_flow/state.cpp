#include "sdfg/control_flow/state.h"

#include "sdfg/sdfg.h"

namespace sdfg {
namespace control_flow {

State::State(size_t element_id, const DebugInfo& debug_info, const graph::Vertex vertex)
    : Element(element_id, debug_info), vertex_(vertex) {
    this->dataflow_ = std::make_unique<data_flow::DataFlowGraph>();
};

void State::validate(const Function& function) const { this->dataflow_->validate(function); };

graph::Vertex State::vertex() const { return this->vertex_; };

const data_flow::DataFlowGraph& State::dataflow() const { return *this->dataflow_; };

data_flow::DataFlowGraph& State::dataflow() { return *this->dataflow_; };

void State::replace(const symbolic::Expression& old_expression, const symbolic::Expression& new_expression) {
    this->dataflow_->replace(old_expression, new_expression);
};

ReturnState::ReturnState(
    size_t element_id, const DebugInfo& debug_info, const graph::Vertex vertex, const std::string& data, bool unreachable
)
    : State(element_id, debug_info, vertex), data_(data), unreachable_(unreachable) {};

const std::string& ReturnState::data() const { return this->data_; };

bool ReturnState::unreachable() const { return this->unreachable_; };

void ReturnState::validate(const Function& function) const {
    State::validate(function);

    auto& sdfg = static_cast<const SDFG&>(function);
    if (sdfg.out_degree(*this) > 0) {
        throw InvalidSDFGException("ReturnState must not have outgoing transitions");
    }
}

} // namespace control_flow
} // namespace sdfg
