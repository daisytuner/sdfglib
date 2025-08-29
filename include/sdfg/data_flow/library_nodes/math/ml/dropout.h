#pragma once

#include "sdfg/data_flow/library_nodes/math/math_node.h"

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/serializer/json_serializer.h"

namespace sdfg {
namespace math {
namespace ml {

inline data_flow::LibraryNodeCode LibraryNodeType_Dropout("ml::Dropout");

// Non-training dropout node
class DropoutNode : public math::MathNode {
public:
    DropoutNode(
        size_t element_id,
        const DebugInfoRegion& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent
    );

    void validate(const Function& function) const override;

    bool expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;
};

class DropoutSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override {
        const DropoutNode& elem_node = static_cast<const DropoutNode&>(library_node);
        nlohmann::json j;

        j["code"] = elem_node.code().value();

        return j;
    }

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override {
        // Assertions for required fields
        assert(j.contains("element_id"));
        assert(j.contains("code"));
        assert(j.contains("debug_info"));

        auto code = j["code"].get<std::string>();

        // Extract debug info using JSONSerializer
        sdfg::serializer::JSONSerializer serializer;
        DebugInfoRegion debug_info = serializer.json_to_debug_info_region(j["debug_info"], builder.debug_info());

        return builder.add_library_node<DropoutNode>(parent, debug_info);
    }
};

} // namespace ml
} // namespace math
} // namespace sdfg
