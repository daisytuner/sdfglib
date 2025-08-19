#pragma once

#include "sdfg/data_flow/library_nodes/math/ml/element_wise.h"

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/serializer/json_serializer.h"

namespace sdfg {
namespace math {
namespace ml {

inline data_flow::LibraryNodeCode LibraryNodeType_Sigmoid("Sigmoid");

class SigmoidNode : public ElementWiseUnaryNode {
public:
    SigmoidNode(
        size_t element_id, const DebugInfo& debug_info, const graph::Vertex vertex, data_flow::DataFlowGraph& parent
    );

    bool expand_operation(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::Sequence& body,
        const std::string& input_name,
        const std::string& output_name,
        const types::IType& input_type,
        const types::IType& output_type,
        const data_flow::Subset& subset
    ) override;

    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;
};

typedef ElementWiseUnaryNodeSerializer<SigmoidNode> SigmoidNodeSerializer;

} // namespace ml
} // namespace math
} // namespace sdfg
