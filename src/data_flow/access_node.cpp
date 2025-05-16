#include "sdfg/data_flow/access_node.h"

namespace sdfg {
namespace data_flow {

AccessNode::AccessNode(size_t element_id, const DebugInfo& debug_info, const graph::Vertex vertex,
                       DataFlowGraph& parent, const std::string& data)
    : DataFlowNode(element_id, debug_info, vertex, parent), data_(data) {

      };

const std::string& AccessNode::data() const { return this->data_; };

std::string& AccessNode::data() { return this->data_; };

std::unique_ptr<DataFlowNode> AccessNode::clone(const graph::Vertex vertex,
                                                DataFlowGraph& parent) const {
    return std::unique_ptr<AccessNode>(
        new AccessNode(this->element_id_, this->debug_info_, vertex, parent, this->data_));
};

void AccessNode::replace(const symbolic::Expression& old_expression,
                         const symbolic::Expression& new_expression) {
    if (SymEngine::is_a<SymEngine::Symbol>(*old_expression)) {
        auto old_symbol = SymEngine::rcp_static_cast<const SymEngine::Symbol>(old_expression);
        if (this->data_ == old_symbol->get_name()) {
            assert(SymEngine::is_a<SymEngine::Symbol>(*new_expression) &&
                   "Access Nodes do not support complex expressions");
            auto new_symbol = SymEngine::rcp_static_cast<const SymEngine::Symbol>(new_expression);
            this->data_ = new_symbol->get_name();
        }
    }
};

}  // namespace data_flow
}  // namespace sdfg
