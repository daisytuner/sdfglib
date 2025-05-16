#include "sdfg/data_flow/code_node.h"

namespace sdfg {
namespace data_flow {

CodeNode::CodeNode(size_t element_id, const DebugInfo& debug_info, const graph::Vertex vertex,
                   DataFlowGraph& parent,
                   const std::vector<std::pair<std::string, sdfg::types::Scalar>>& outputs,
                   const std::vector<std::pair<std::string, sdfg::types::Scalar>>& inputs)
    : DataFlowNode(element_id, debug_info, vertex, parent), outputs_(outputs), inputs_(inputs) {

      };

const std::vector<std::pair<std::string, sdfg::types::Scalar>>& CodeNode::outputs() const {
    return this->outputs_;
};

const std::pair<std::string, sdfg::types::Scalar> CodeNode::output(size_t index) const {
    return this->outputs_[index];
};

const sdfg::types::Scalar& CodeNode::output_type(const std::string& input) const {
    return this->outputs_.at(0).second;
};

const std::vector<std::pair<std::string, sdfg::types::Scalar>>& CodeNode::inputs() const {
    return this->inputs_;
};

const std::pair<std::string, sdfg::types::Scalar> CodeNode::input(size_t index) const {
    return this->inputs_[index];
};

const sdfg::types::Scalar& CodeNode::input_type(const std::string& input) const {
    return std::find_if(this->inputs_.begin(), this->inputs_.end(),
                        [&input](const std::pair<std::string, sdfg::types::Scalar>& p) {
                            return p.first == input;
                        })
        ->second;
};

bool CodeNode::needs_connector(size_t index) const {
    // Is non-constant, if starts with _in prefix
    if (this->inputs_[index].first.compare(0, 3, "_in") == 0) {
        return true;
    }
    return false;
};

}  // namespace data_flow
}  // namespace sdfg
