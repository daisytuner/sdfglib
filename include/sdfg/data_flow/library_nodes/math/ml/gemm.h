#pragma once

#include "sdfg/data_flow/library_nodes/math/math_node.h"

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/serializer/json_serializer.h"

namespace sdfg {
namespace math {
namespace ml {

inline data_flow::LibraryNodeCode LibraryNodeType_Gemm("ml::Gemm");

class GemmNode : public MathNode {
private:
    std::string alpha_;
    std::string beta_;
    bool trans_a_;
    bool trans_b_;

public:
    GemmNode(
        size_t element_id,
        const DebugInfo &debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph &parent,
        const std::string &alpha = "1.0f",
        const std::string &beta = "1.0f",
        bool trans_a = false,
        bool trans_b = false
    );

    const std::string &alpha() const { return alpha_; }
    const std::string &beta() const { return beta_; }
    bool trans_a() const { return trans_a_; }
    bool trans_b() const { return trans_b_; }

    void validate(const Function &function) const override;

    bool expand(builder::StructuredSDFGBuilder &builder, analysis::AnalysisManager &analysis_manager) override;

    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph &parent) const override;
};

class GemmNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode &library_node) override;

    data_flow::LibraryNode &deserialize(
        const nlohmann::json &j, builder::StructuredSDFGBuilder &builder, structured_control_flow::Block &parent
    ) override;
};

}
}
}
