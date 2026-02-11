#pragma once

#include "sdfg/data_flow/library_nodes/math/tensor/tensor_node.h"

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/serializer/json_serializer.h"

namespace sdfg {
namespace math {
namespace tensor {

inline data_flow::LibraryNodeCode LibraryNodeType_Conv("ml::Conv");

class ConvNode : public TensorNode {
protected:
    std::vector<symbolic::Expression> kernel_shape_;
    std::vector<symbolic::Expression> strides_;
    std::vector<symbolic::Expression> pads_;
    std::vector<symbolic::Expression> dilations_;
    symbolic::Expression output_channels_;
    symbolic::Expression group_;

public:
    ConvNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const std::vector<symbolic::Expression>& kernel_shape,
        const std::vector<symbolic::Expression>& strides,
        const std::vector<symbolic::Expression>& pads,
        const std::vector<symbolic::Expression>& dilations,
        symbolic::Expression output_channels,
        symbolic::Expression group
    );

    const std::vector<symbolic::Expression>& kernel_shape() const { return kernel_shape_; }

    const std::vector<symbolic::Expression>& strides() const { return strides_; }

    const std::vector<symbolic::Expression>& pads() const { return pads_; }

    const std::vector<symbolic::Expression>& dilations() const { return dilations_; }

    symbolic::Expression output_channels() const { return output_channels_; }

    symbolic::Expression group() const { return group_; }

    void validate(const Function& function) const override;

    bool expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    symbolic::SymbolSet symbols() const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    bool supports_integer_types() const override { return false; }

    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;

    std::string toStr() const override;
};

class ConvNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

} // namespace tensor
} // namespace math
} // namespace sdfg
