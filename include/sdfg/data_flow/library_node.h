#pragma once

#include <vector>

#include "sdfg/data_flow/code_node.h"
#include "sdfg/graph/graph.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {

namespace builder {
class SDFGBuilder;
class StructuredSDFGBuilder;
}  // namespace builder

namespace data_flow {

enum LibraryNodeType {
    LocalBarrier,
    /* GlobalBarrier,
    InterNodeSend,
    InterNodeReceive,
    DeviceAllocation,
    DeviceDeallocation,
    DeviceSend,
    DeviceReceive, */
};

class LibraryNode : public CodeNode {
    friend class sdfg::builder::SDFGBuilder;
    friend class sdfg::builder::StructuredSDFGBuilder;

   private:
    LibraryNodeType call_;
    bool side_effect_;

    LibraryNode(size_t element_id, const DebugInfo& debug_info, const graph::Vertex vertex,
                DataFlowGraph& parent,
                const std::vector<std::pair<std::string, sdfg::types::Scalar>>& outputs,
                const std::vector<std::pair<std::string, sdfg::types::Scalar>>& inputs,
                const LibraryNodeType& call, const bool has_side_effect);

   public:
    LibraryNode(const LibraryNode& data_node) = delete;
    LibraryNode& operator=(const LibraryNode&) = delete;

    const std::vector<std::string>& params() const;

    const LibraryNodeType& call() const;

    virtual std::unique_ptr<DataFlowNode> clone(const graph::Vertex vertex,
                                                DataFlowGraph& parent) const override;

    void replace(const symbolic::Expression& old_expression,
                 const symbolic::Expression& new_expression) override;

    bool has_side_effect() const;
};

}  // namespace data_flow
}  // namespace sdfg
