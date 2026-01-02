#pragma once

#include "sdfg/data_flow/library_nodes/math/tensor/reduce_node.h"

namespace sdfg {
namespace math {
namespace tensor {

inline data_flow::LibraryNodeCode LibraryNodeType_Softmax("ml::Softmax");

class SoftmaxNode : public ReduceNode {
public:
    SoftmaxNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const std::vector<symbolic::Expression>& shape,
        const std::vector<int64_t>& axes,
        bool keepdims = false
    );

    bool expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    // Not used but required by abstract base class if I didn't override expand?
    // Actually ReduceNode::expand calls expand_reduction, but since I override expand, I don't need to implement
    // expand_reduction if I don't call ReduceNode::expand. However, ReduceNode declares expand_reduction as pure
    // virtual? No, it's virtual but not pure in the header I read? Let's check ReduceNode header again. It says:
    // virtual bool expand_reduction(...) = 0; So I MUST implement it, even if I don't use it.
    bool expand_reduction(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::Sequence& body,
        const std::string& input_name,
        const std::string& output_name,
        const types::IType& input_type,
        const types::IType& output_type,
        const data_flow::Subset& input_subset,
        const data_flow::Subset& output_subset
    ) override {
        return false;
    }

    std::string identity() const override { return ""; }

    bool supports_integer_types() const override { return false; }

    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;
};

typedef ReduceNodeSerializer<SoftmaxNode> SoftmaxNodeSerializer;

} // namespace tensor
} // namespace math
} // namespace sdfg
