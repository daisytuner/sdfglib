#pragma once

#include "sdfg/data_flow/data_flow_node.h"
#include "sdfg/graph/graph.h"
#include "sdfg/types/scalar.h"

namespace sdfg {

namespace builder {
class SDFGBuilder;
class StructuredSDFGBuilder;
}  // namespace builder

namespace data_flow {

class CodeNode : public DataFlowNode {
    friend class sdfg::builder::SDFGBuilder;
    friend class sdfg::builder::StructuredSDFGBuilder;

   protected:
    CodeNode(const DebugInfo& debug_info, const graph::Vertex vertex, DataFlowGraph& parent);

   public:
    CodeNode(const CodeNode& data_node) = delete;
    CodeNode& operator=(const CodeNode&) = delete;

    virtual bool needs_connector(size_t index) const = 0;

    virtual std::unique_ptr<DataFlowNode> clone(const graph::Vertex vertex,
                                                DataFlowGraph& parent) const = 0;
};
}  // namespace data_flow
}  // namespace sdfg