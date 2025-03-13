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
    std::vector<std::pair<std::string, sdfg::types::Scalar>> outputs_;
    std::vector<std::pair<std::string, sdfg::types::Scalar>> inputs_;

    CodeNode(size_t element_id, const DebugInfo& debug_info, const graph::Vertex& vertex,
             DataFlowGraph& parent,
             const std::vector<std::pair<std::string, sdfg::types::Scalar>>& outputs,
             const std::vector<std::pair<std::string, sdfg::types::Scalar>>& inputs);

   public:
    CodeNode(const CodeNode& data_node) = delete;
    CodeNode& operator=(const CodeNode&) = delete;

    const std::vector<std::pair<std::string, sdfg::types::Scalar>>& outputs() const;

    const std::pair<std::string, sdfg::types::Scalar> output(size_t index) const;

    const sdfg::types::Scalar& output_type(const std::string& output) const;

    const std::vector<std::pair<std::string, sdfg::types::Scalar>>& inputs() const;

    const std::pair<std::string, sdfg::types::Scalar> input(size_t index) const;

    const sdfg::types::Scalar& input_type(const std::string& input) const;

    bool needs_connector(size_t index) const;

    virtual std::unique_ptr<DataFlowNode> clone(const graph::Vertex& vertex,
                                                DataFlowGraph& parent) const = 0;
};
}  // namespace data_flow
}  // namespace sdfg