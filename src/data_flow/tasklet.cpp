#include "sdfg/data_flow/tasklet.h"

#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace data_flow {

Tasklet::Tasklet(size_t element_id, const DebugInfo& debug_info, const graph::Vertex vertex,
                 DataFlowGraph& parent, const TaskletCode code,
                 const std::pair<std::string, sdfg::types::Scalar>& output,
                 const std::vector<std::pair<std::string, sdfg::types::Scalar>>& inputs,
                 const symbolic::Condition& condition)
    : CodeNode(element_id, debug_info, vertex, parent),
      code_(code),
      output_(output),
      inputs_(inputs),
      condition_(condition) {};

TaskletCode Tasklet::code() const { return this->code_; };

const std::vector<std::pair<std::string, sdfg::types::Scalar>>& Tasklet::inputs() const {
    return this->inputs_;
};

const std::pair<std::string, sdfg::types::Scalar>& Tasklet::output() const {
    return this->output_;
};

const std::pair<std::string, sdfg::types::Scalar>& Tasklet::input(size_t index) const {
    return this->inputs_[index];
};

const sdfg::types::Scalar& Tasklet::input_type(const std::string& input) const {
    return std::find_if(this->inputs_.begin(), this->inputs_.end(),
                        [&input](const std::pair<std::string, sdfg::types::Scalar>& p) {
                            return p.first == input;
                        })
        ->second;
};

bool Tasklet::needs_connector(size_t index) const {
    // Is non-constant, if starts with _in prefix
    if (this->inputs_[index].first.compare(0, 3, "_in") == 0) {
        return true;
    }
    return false;
};

const sdfg::types::Scalar& Tasklet::output_type() const { return this->output_.second; };

const symbolic::Condition& Tasklet::condition() const { return this->condition_; };

symbolic::Condition& Tasklet::condition() { return this->condition_; };

bool Tasklet::is_conditional() const { return !symbolic::is_true(this->condition_); };

std::unique_ptr<DataFlowNode> Tasklet::clone(size_t element_id, const graph::Vertex vertex,
                                             DataFlowGraph& parent) const {
    return std::unique_ptr<Tasklet>(new Tasklet(element_id, this->debug_info_, vertex, parent,
                                                this->code_, this->output_, this->inputs_,
                                                this->condition_));
};

void Tasklet::replace(const symbolic::Expression& old_expression,
                      const symbolic::Expression& new_expression) {
    this->condition_ = symbolic::subs(this->condition_, old_expression, new_expression);
};

}  // namespace data_flow
}  // namespace sdfg
