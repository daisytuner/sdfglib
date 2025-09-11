#include "sdfg/data_flow/access_node.h"

#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/function.h"

namespace sdfg {
namespace data_flow {

AccessNode::AccessNode(
    size_t element_id,
    const DebugInfoRegion& debug_info_region,
    const graph::Vertex vertex,
    DataFlowGraph& parent,
    const std::string& data
)
    : DataFlowNode(element_id, debug_info_region, vertex, parent), data_(data) {

      };

void AccessNode::validate(const Function& function) const {
    auto& graph = this->get_parent();

    if (graph.out_degree(*this) > 1) {
        MemletType type = (*graph.out_edges(*this).begin()).type();
        for (auto& oedge : graph.out_edges(*this)) {
            if (oedge.type() != type) {
                throw InvalidSDFGException("Access node " + this->data() + " used with multiple memlet types");
            }
        }
    }

    if (graph.in_degree(*this) > 1) {
        MemletType type = (*graph.in_edges(*this).begin()).type();
        for (auto& iedge : graph.in_edges(*this)) {
            if (iedge.type() != type) {
                throw InvalidSDFGException("Access node " + this->data() + " used with multiple memlet types");
            }
        }
    }
}

const std::string& AccessNode::data() const { return this->data_; };

std::string& AccessNode::data() { return this->data_; };

std::unique_ptr<DataFlowNode> AccessNode::clone(size_t element_id, const graph::Vertex vertex, DataFlowGraph& parent)
    const {
    return std::unique_ptr<AccessNode>(new AccessNode(element_id, this->debug_info_, vertex, parent, this->data_));
};

void AccessNode::replace(const symbolic::Expression& old_expression, const symbolic::Expression& new_expression) {
    if (SymEngine::is_a<SymEngine::Symbol>(*old_expression) && SymEngine::is_a<SymEngine::Symbol>(*new_expression)) {
        auto old_symbol = SymEngine::rcp_static_cast<const SymEngine::Symbol>(old_expression);
        if (this->data_ == old_symbol->get_name()) {
            auto new_symbol = SymEngine::rcp_static_cast<const SymEngine::Symbol>(new_expression);
            this->data_ = new_symbol->get_name();
        }
    }
};

} // namespace data_flow
} // namespace sdfg
