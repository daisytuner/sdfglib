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
} // namespace builder

namespace data_flow {

class DataFlowGraph;

typedef std::vector<symbolic::Expression> Subset;

class Memlet : public Element {
    friend class sdfg::builder::SDFGBuilder;
    friend class sdfg::builder::StructuredSDFGBuilder;

private:
    // Remark: Exclusive resource
    const graph::Edge edge_;

    DataFlowGraph* parent_;

    DataFlowNode& src_;
    DataFlowNode& dst_;
    std::string src_conn_;
    std::string dst_conn_;
    Subset begin_subset_;
    Subset end_subset_;

    Memlet(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Edge& edge,
        DataFlowGraph& parent,
        DataFlowNode& src,
        const std::string& src_conn,
        DataFlowNode& dst,
        const std::string& dst_conn,
        const Subset& subset
    );

    Memlet(
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
    );

public:
    // Remark: Exclusive resource
    Memlet(const Memlet& memlet) = delete;
    Memlet& operator=(const Memlet&) = delete;

    const graph::Edge edge() const;

    const DataFlowGraph& get_parent() const;

    DataFlowGraph& get_parent();

    const DataFlowNode& src() const;

    DataFlowNode& src();

    const DataFlowNode& dst() const;

    DataFlowNode& dst();

    const std::string& src_conn() const;

    const std::string& dst_conn() const;

    const Subset subset() const;

    void set_subset(const Subset& subset);

    const Subset begin_subset() const;

    const Subset end_subset() const;

    bool has_range() const;

    void set_subset(const Subset& begin_subset, const Subset& end_subset);

    std::unique_ptr<Memlet> clone(
        size_t element_id, const graph::Edge& edge, DataFlowGraph& parent, DataFlowNode& src, DataFlowNode& dst
    ) const;

    void replace(const symbolic::Expression& old_expression, const symbolic::Expression& new_expression) override;
};
} // namespace data_flow
} // namespace sdfg
