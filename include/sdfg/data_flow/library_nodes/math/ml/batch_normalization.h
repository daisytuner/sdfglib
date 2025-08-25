#pragma once

#include "sdfg/data_flow/library_nodes/math/math_node.h"

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/serializer/json_serializer.h"

namespace sdfg {
namespace math {
namespace ml {

inline data_flow::LibraryNodeCode LibraryNodeType_BatchNormalization("ml::BatchNormalization");

class BatchNormalizationNode : public MathNode {
private:
    int axis_;
    std::string epsilon_;

public:
    BatchNormalizationNode(size_t element_id, const DebugInfo& debug_info, const graph::Vertex vertex, data_flow::DataFlowGraph& parent, int axis = -1, const std::string& epsilon = "0.00001f");

    int axis() const { return axis_; }

    const std::string& epsilon() const { return epsilon_; }

    void validate(const Function& function) const override;

    bool expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;
};

class BatchNormalizationNodeSerializer : public serializer::LibraryNodeSerializer {
    public:
        nlohmann::json serialize(const data_flow::LibraryNode &library_node) override;
    
        data_flow::LibraryNode &deserialize(
            const nlohmann::json &j, builder::StructuredSDFGBuilder &builder, structured_control_flow::Block &parent
        ) override;
};

} // namespace ml
} // namespace math
} // namespace sdfg
