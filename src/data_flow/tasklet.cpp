#include "sdfg/data_flow/tasklet.h"

#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace data_flow {

Tasklet::Tasklet(
    size_t element_id,
    const DebugInfoRegion& debug_info,
    const graph::Vertex vertex,
    DataFlowGraph& parent,
    const TaskletCode code,
    const std::string& output,
    const std::vector<std::string>& inputs
)
    : CodeNode(element_id, debug_info, vertex, parent, {output}, inputs), code_(code) {};

void Tasklet::validate(const Function& function) const {
    if (arity(this->code_) != this->inputs_.size()) {
        throw InvalidSDFGException(
            "Tasklet: Invalid number of inputs for code " + std::to_string(this->code_) + ": expected " +
            std::to_string(arity(this->code_)) + ", got " + std::to_string(this->inputs_.size())
        );
    }

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

std::unique_ptr<DataFlowNode> Tasklet::clone(size_t element_id, const graph::Vertex vertex, DataFlowGraph& parent)
    const {
    return std::unique_ptr<Tasklet>(
        new Tasklet(element_id, this->debug_info_, vertex, parent, this->code_, this->outputs_.at(0), this->inputs_)
    );
};

void Tasklet::replace(const symbolic::Expression& old_expression, const symbolic::Expression& new_expression) {};

} // namespace data_flow
} // namespace sdfg
