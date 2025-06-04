#pragma once

#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/data_flow_node.h"
#include "sdfg/element.h"
#include "sdfg/graph/graph.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {

namespace builder {
class SDFGBuilder;
class StructuredSDFGBuilder;
}  // namespace builder

namespace data_flow {

class DataFlowGraph;

typedef std::vector<symbolic::Expression> Subset;

class Memlet : public Element {
    friend class sdfg::builder::SDFGBuilder;
    friend class sdfg::builder::StructuredSDFGBuilder;

   private:
    // Remark: Exclusive resource
    const graph::Edge edge_;

    const DataFlowGraph* parent_;

    DataFlowNode& src_;
    DataFlowNode& dst_;
    std::string src_conn_;
    std::string dst_conn_;
    Subset subset_;

    Memlet(const DebugInfo& debug_info, const graph::Edge& edge, const DataFlowGraph& parent,
           DataFlowNode& src, const std::string& src_conn, DataFlowNode& dst,
           const std::string& dst_conn, const Subset& subset);

   public:
    // Remark: Exclusive resource
    Memlet(const Memlet& memlet) = delete;
    Memlet& operator=(const Memlet&) = delete;

    const graph::Edge edge() const;

    const DataFlowGraph& get_parent() const;

    const DataFlowNode& src() const;

    DataFlowNode& src();

    const DataFlowNode& dst() const;

    DataFlowNode& dst();

    const std::string& src_conn() const;

    const std::string& dst_conn() const;

    const Subset subset() const;

    Subset& subset();

    std::unique_ptr<Memlet> clone(const graph::Edge& edge, const DataFlowGraph& parent,
                                  DataFlowNode& src, DataFlowNode& dst) const;

    void replace(const symbolic::Expression& old_expression,
                 const symbolic::Expression& new_expression) override;
};
}  // namespace data_flow
}  // namespace sdfg