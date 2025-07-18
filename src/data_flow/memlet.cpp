#include <sdfg/data_flow/memlet.h>

#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace data_flow {

Memlet::Memlet(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Edge& edge,
    DataFlowGraph& parent,
    DataFlowNode& src,
    const std::string& src_conn,
    DataFlowNode& dst,
    const std::string& dst_conn,
    const Subset& subset
)
    : Element(element_id, debug_info), edge_(edge), parent_(&parent), src_(src), dst_(dst), src_conn_(src_conn),
      dst_conn_(dst_conn), begin_subset_(subset), end_subset_(subset) {

      };

Memlet::Memlet(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Edge& edge,
    DataFlowGraph& parent,
    DataFlowNode& src,
    const std::string& src_conn,
    DataFlowNode& dst,
    const std::string& dst_conn,
    const Subset& begin_subset,
    const Subset& end_subset
)
    : Element(element_id, debug_info), edge_(edge), parent_(&parent), src_(src), dst_(dst), src_conn_(src_conn),
      dst_conn_(dst_conn), begin_subset_(begin_subset), end_subset_(end_subset) {

      };

void Memlet::validate() const {
    // TODO: Implement
};

const graph::Edge Memlet::edge() const { return this->edge_; };

const DataFlowGraph& Memlet::get_parent() const { return *this->parent_; };

DataFlowGraph& Memlet::get_parent() { return *this->parent_; };

const DataFlowNode& Memlet::src() const { return this->src_; };

DataFlowNode& Memlet::src() { return this->src_; };

const DataFlowNode& Memlet::dst() const { return this->dst_; };

DataFlowNode& Memlet::dst() { return this->dst_; };

const std::string& Memlet::src_conn() const { return this->src_conn_; };

const std::string& Memlet::dst_conn() const { return this->dst_conn_; };

const Subset Memlet::subset() const { return this->begin_subset_; };

void Memlet::set_subset(const Subset& subset) {
    this->begin_subset_ = subset;
    this->end_subset_ = subset;
};

const Subset Memlet::begin_subset() const { return this->begin_subset_; };

const Subset Memlet::end_subset() const { return this->end_subset_; };

void Memlet::set_subset(const Subset& begin_subset, const Subset& end_subset) {
    this->begin_subset_ = begin_subset;
    this->end_subset_ = end_subset;
};

bool Memlet::has_range() const {
    for (size_t i = 0; i < this->begin_subset_.size(); i++) {
        if (!symbolic::eq(this->begin_subset_[i], this->end_subset_[i])) {
            return true;
        }
    }
    return true;
};

std::unique_ptr<Memlet> Memlet::clone(
    size_t element_id, const graph::Edge& edge, DataFlowGraph& parent, DataFlowNode& src, DataFlowNode& dst
) const {
    return std::unique_ptr<Memlet>(new Memlet(
        element_id,
        this->debug_info_,
        edge,
        parent,
        src,
        this->src_conn_,
        dst,
        this->dst_conn_,
        this->begin_subset_,
        this->end_subset_
    ));
};

void Memlet::replace(const symbolic::Expression& old_expression, const symbolic::Expression& new_expression) {
    for (auto& dim : this->begin_subset_) {
        dim = symbolic::subs(dim, old_expression, new_expression);
    }
    for (auto& dim : this->end_subset_) {
        dim = symbolic::subs(dim, old_expression, new_expression);
    }
};

} // namespace data_flow
} // namespace sdfg
