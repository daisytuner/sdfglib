#include "sdfg/data_flow/code_node.h"

#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/data_flow_graph.h"

namespace sdfg {
namespace data_flow {

CodeNode::CodeNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    DataFlowGraph& parent,
    const std::vector<std::string>& outputs,
    const std::vector<std::string>& inputs
)
    : DataFlowNode(element_id, debug_info, vertex, parent), outputs_(outputs), inputs_(inputs) {};

const std::vector<std::string>& CodeNode::outputs() const { return this->outputs_; };

const std::vector<std::string>& CodeNode::inputs() const { return this->inputs_; };

std::vector<std::string>& CodeNode::outputs() { return this->outputs_; };

std::vector<std::string>& CodeNode::inputs() { return this->inputs_; };

const std::string& CodeNode::output(size_t index) const { return this->outputs_[index]; };

const std::string& CodeNode::input(size_t index) const { return this->inputs_[index]; };

bool CodeNode::has_constant_input(size_t index) const {
    for (auto& iedge : this->get_parent().in_edges(*this)) {
        if (iedge.dst_conn() == this->inputs_[index]) {
            if (dynamic_cast<const ConstantNode*>(&iedge.src())) {
                return true;
            }
        }
    }

    return false;
}

void CodeNode::mark_input_optional(const std::string& connector_name) {
    optional_inputs_.insert(connector_name);
}

bool CodeNode::is_input_optional(const std::string& connector_name) const {
    return optional_inputs_.find(connector_name) != optional_inputs_.end();
}

} // namespace data_flow
} // namespace sdfg
