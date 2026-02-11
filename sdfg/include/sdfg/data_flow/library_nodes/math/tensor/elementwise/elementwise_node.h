#pragma once

#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_node.h"
#include "sdfg/data_flow/tasklet.h"

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/serializer/json_serializer.h"

namespace sdfg {
namespace math {
namespace tensor {

inline data_flow::LibraryNodeCode LibraryNodeType_Elementwise("ml::Elementwise");

class ElementwiseNode : public TensorNode {
private:
    // Tasklet code-based
    std::optional<data_flow::TaskletCode> tasklet_code_;

    // CMath function-based
    std::optional<math::cmath::CMathFunction> cmath_function_;
    std::optional<types::PrimitiveType> precision_;

public:
    ElementwiseNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const data_flow::TaskletCode& code
    );

    ElementwiseNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const math::cmath::CMathFunction& function,
        types::PrimitiveType precision
    );

    std::optional<data_flow::TaskletCode> tasklet_code() const { return tasklet_code_; }

    std::optional<math::cmath::CMathFunction> cmath_function() const { return cmath_function_; }

    std::optional<types::PrimitiveType> precision() const { return precision_; }

    bool supports_integer_types() const override { return true; }

    symbolic::SymbolSet symbols() const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    bool expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    std::unique_ptr<DataFlowNode> clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent)
        const override;
};

class ElementwiseNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

} // namespace tensor
} // namespace math
} // namespace sdfg
