#pragma once

#include "sdfg/data_flow/library_nodes/math/math_node.h"

#include "sdfg/debug_info.h"
#include "sdfg/serializer/json_serializer.h"

namespace sdfg {
namespace math {
namespace ml {

inline data_flow::LibraryNodeCode LibraryNodeType_MaxPool("ml::MaxPool");

class MaxPoolNode : public MathNode {
private:
    std::vector<size_t> kernel_shape_;
    std::vector<size_t> pads_;
    std::vector<size_t> strides_;

public:
    MaxPoolNode(
        size_t element_id,
        const DebugInfoRegion &debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph &parent,
        std::vector<size_t> kernel_shape,
        std::vector<size_t> pads,
        std::vector<size_t> strides
    );

    std::vector<size_t> kernel_shape() const;
    std::vector<size_t> pads() const;
    std::vector<size_t> strides() const;

    void validate(const Function &function) const override;

    bool expand(builder::StructuredSDFGBuilder &builder, analysis::AnalysisManager &analysis_manager) override;

    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph &parent) const override;
};

class MaxPoolNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode &library_node) override;

    data_flow::LibraryNode &deserialize(
        const nlohmann::json &j, builder::StructuredSDFGBuilder &builder, structured_control_flow::Block &parent
    ) override;
};

} // namespace ml
} // namespace math
} // namespace sdfg
