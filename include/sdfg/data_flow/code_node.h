#pragma once

#include "sdfg/data_flow/data_flow_node.h"
#include "sdfg/graph/graph.h"
#include "sdfg/types/scalar.h"

namespace sdfg {

namespace builder {
class SDFGBuilder;
class StructuredSDFGBuilder;
} // namespace builder

namespace data_flow {

class CodeNode : public DataFlowNode {
    friend class sdfg::builder::SDFGBuilder;
    friend class sdfg::builder::StructuredSDFGBuilder;

protected:
    std::vector<std::string> outputs_;
    std::vector<std::string> inputs_;

    CodeNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        DataFlowGraph& parent,
        const std::vector<std::string>& outputs,
        const std::vector<std::string>& inputs
    );

public:
    CodeNode(const CodeNode& data_node) = delete;
    CodeNode& operator=(const CodeNode&) = delete;

    const std::vector<std::string>& outputs() const;

    const std::vector<std::string>& inputs() const;

    std::vector<std::string>& outputs();

    std::vector<std::string>& inputs();

    const std::string& output(size_t index) const;

    const std::string& input(size_t index) const;

    virtual bool needs_connector(size_t index) const = 0;

    virtual std::unique_ptr<DataFlowNode> clone(size_t element_id, const graph::Vertex vertex, DataFlowGraph& parent)
        const = 0;
};
} // namespace data_flow
} // namespace sdfg
