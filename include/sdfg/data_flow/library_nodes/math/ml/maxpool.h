#pragma once

#include "sdfg/data_flow/library_nodes/math/math_node.h"

namespace sdfg {
namespace math {
namespace ml {

inline data_flow::LibraryNodeCode LibraryNodeType_MaxPool("MaxPool");

class MaxPoolNode : public math::MathNode {
private:
    std::vector<int> dilations_;
    std::vector<int> kernel_shape_;
    std::vector<int> pads_;
    std::vector<int> strides_;

public:
    MaxPoolNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const std::string& output,
        const std::string& input,
        const std::vector<int>& dilations,
        const std::vector<int>& kernel_shape,
        const std::vector<int>& pads,
        const std::vector<int>& strides
    );

    bool expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;
};

} // namespace ml
} // namespace math
} // namespace sdfg
