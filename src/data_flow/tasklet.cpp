#include "sdfg/data_flow/tasklet.h"

#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace data_flow {

Tasklet::Tasklet(size_t element_id, const DebugInfo& debug_info, const graph::Vertex vertex,
                 DataFlowGraph& parent, const TaskletCode code,
                 const std::pair<std::string, sdfg::types::Scalar>& output,
                 const std::vector<std::pair<std::string, sdfg::types::Scalar>>& inputs,
                 const symbolic::Condition& condition)
    : CodeNode(element_id, debug_info, vertex, parent, {output}, inputs),
      code_(code),
      condition_(condition) {};

const TaskletCode Tasklet::code() const { return this->code_; };

const symbolic::Condition& Tasklet::condition() const { return this->condition_; };

symbolic::Condition& Tasklet::condition() { return this->condition_; };

bool Tasklet::is_conditional() const { return !symbolic::is_true(this->condition_); };

std::unique_ptr<DataFlowNode> Tasklet::clone(const graph::Vertex vertex,
                                             DataFlowGraph& parent) const {
    return std::unique_ptr<Tasklet>(new Tasklet(this->element_id_, this->debug_info_, vertex,
                                                parent, this->code_, this->outputs_.at(0),
                                                this->inputs_, this->condition_));
};

void Tasklet::replace(const symbolic::Expression& old_expression,
                      const symbolic::Expression& new_expression) {
    this->condition_ = symbolic::subs(this->condition_, old_expression, new_expression);
};

}  // namespace data_flow
}  // namespace sdfg
