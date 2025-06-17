#pragma once

#include <vector>

#include "sdfg/data_flow/library_node.h"

namespace sdfg {

namespace builder {
class SDFGBuilder;
class StructuredSDFGBuilder;
}  // namespace builder

namespace data_flow {

class BarrierLocalNode : public LibraryNode {
    friend class sdfg::builder::SDFGBuilder;
    friend class sdfg::builder::StructuredSDFGBuilder;

   protected:
    BarrierLocalNode(size_t element_id, const DebugInfo& debug_info, const graph::Vertex vertex,
                     DataFlowGraph& parent, const data_flow::LibraryNodeCode code,
                     const std::vector<std::string>& outputs,
                     const std::vector<std::string>& inputs, const bool side_effect = true);

   public:
    BarrierLocalNode(const BarrierLocalNode& data_node) = delete;
    BarrierLocalNode& operator=(const BarrierLocalNode&) = delete;

    virtual ~BarrierLocalNode() = default;

    const LibraryNodeCode& code() const;

    const std::vector<std::string>& inputs() const;

    const std::vector<std::string>& outputs() const;

    const std::string& input(size_t index) const;

    const std::string& output(size_t index) const;

    bool side_effect() const;

    bool needs_connector(size_t index) const override;

    std::unique_ptr<DataFlowNode> clone(const graph::Vertex vertex,
                                        DataFlowGraph& parent) const override;

    void replace(const symbolic::Expression& old_expression,
                 const symbolic::Expression& new_expression) override;
};

}  // namespace data_flow
}  // namespace sdfg
