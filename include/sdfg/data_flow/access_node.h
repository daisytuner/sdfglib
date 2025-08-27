#pragma once

#include "sdfg/data_flow/data_flow_node.h"
#include "sdfg/debug_info.h"
#include "sdfg/graph/graph.h"

namespace sdfg {

namespace builder {
class SDFGBuilder;
class StructuredSDFGBuilder;
} // namespace builder

namespace data_flow {

class AccessNode : public DataFlowNode {
    friend class sdfg::builder::SDFGBuilder;
    friend class sdfg::builder::StructuredSDFGBuilder;

private:
    std::string data_;

    AccessNode(
        size_t element_id,
        const DebugInfoRegion& debug_info_region,
        const graph::Vertex vertex,
        DataFlowGraph& parent,
        const std::string& data
    );

public:
    AccessNode(const AccessNode& data_node) = delete;
    AccessNode& operator=(const AccessNode&) = delete;

    void validate(const Function& function) const override;

    const std::string& data() const;

    std::string& data();

    virtual std::unique_ptr<DataFlowNode> clone(size_t element_id, const graph::Vertex vertex, DataFlowGraph& parent)
        const override;

    void replace(const symbolic::Expression& old_expression, const symbolic::Expression& new_expression) override;
};
} // namespace data_flow
} // namespace sdfg
