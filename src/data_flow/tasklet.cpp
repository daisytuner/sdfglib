#include "sdfg/data_flow/tasklet.h"

#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace data_flow {

Tasklet::Tasklet(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    DataFlowGraph& parent,
    const TaskletCode code,
    const std::string& output,
    const std::vector<std::string>& inputs,
    const symbolic::Condition& condition
)
    : CodeNode(element_id, debug_info, vertex, parent, {output}, inputs), code_(code), condition_(condition) {};

void Tasklet::validate(const Function& function) const {
    auto& graph = this->get_parent();

    std::unordered_map<std::string, const AccessNode*> input_names;
    for (auto& iedge : graph.in_edges(*this)) {
        auto& src = static_cast<const AccessNode&>(iedge.src());
        if (input_names.find(src.data()) != input_names.end()) {
            if (input_names.at(src.data()) != &src) {
                throw InvalidSDFGException("Tasklet: Two access nodes with the same data as iedge: " + src.data());
            }
        } else {
            input_names.insert({src.data(), &src});
        }
    }
}

TaskletCode Tasklet::code() const { return this->code_; };

const std::string& Tasklet::output() const { return this->outputs_[0]; };

bool Tasklet::needs_connector(size_t index) const {
    // Is non-constant, if starts with _in prefix
    if (this->inputs_[index].compare(0, 3, "_in") == 0) {
        return true;
    }
    return false;
};

const symbolic::Condition& Tasklet::condition() const { return this->condition_; };

symbolic::Condition& Tasklet::condition() { return this->condition_; };

bool Tasklet::is_conditional() const { return !symbolic::is_true(this->condition_); };

std::unique_ptr<DataFlowNode> Tasklet::clone(size_t element_id, const graph::Vertex vertex, DataFlowGraph& parent)
    const {
    return std::unique_ptr<Tasklet>(new Tasklet(
        element_id, this->debug_info_, vertex, parent, this->code_, this->outputs_.at(0), this->inputs_, this->condition_
    ));
};

void Tasklet::replace(const symbolic::Expression& old_expression, const symbolic::Expression& new_expression) {
    this->condition_ = symbolic::subs(this->condition_, old_expression, new_expression);
};

} // namespace data_flow
} // namespace sdfg
