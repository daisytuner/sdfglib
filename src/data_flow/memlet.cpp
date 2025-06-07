#include <sdfg/data_flow/memlet.h>

#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace data_flow {

Memlet::Memlet(const DebugInfo& debug_info, const graph::Edge& edge, const DataFlowGraph& parent,
               DataFlowNode& src, const std::string& src_conn, DataFlowNode& dst,
               const std::string& dst_conn, const Subset& subset)
    : Element(debug_info),
      edge_(edge),
      parent_(&parent),
      src_(src),
      dst_(dst),
      src_conn_(src_conn),
      dst_conn_(dst_conn),
      subset_(subset) {

      };

const graph::Edge Memlet::edge() const { return this->edge_; };

const DataFlowGraph& Memlet::get_parent() const { return *this->parent_; };

const DataFlowNode& Memlet::src() const { return this->src_; };

DataFlowNode& Memlet::src() { return this->src_; };

const DataFlowNode& Memlet::dst() const { return this->dst_; };

DataFlowNode& Memlet::dst() { return this->dst_; };

const std::string& Memlet::src_conn() const { return this->src_conn_; };

const std::string& Memlet::dst_conn() const { return this->dst_conn_; };

const Subset Memlet::subset() const { return this->subset_; };

Subset& Memlet::subset() { return this->subset_; };

std::unique_ptr<Memlet> Memlet::clone(const graph::Edge& edge, const DataFlowGraph& parent,
                                      DataFlowNode& src, DataFlowNode& dst) const {
    return std::unique_ptr<Memlet>(new Memlet(this->debug_info_, edge, parent, src, this->src_conn_,
                                              dst, this->dst_conn_, this->subset_));
};

void Memlet::replace(const symbolic::Expression& old_expression,
                     const symbolic::Expression& new_expression) {
    for (auto& dim : this->subset_) {
        dim = symbolic::subs(dim, old_expression, new_expression);
    }
};

}  // namespace data_flow
}  // namespace sdfg
