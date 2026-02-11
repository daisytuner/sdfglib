#pragma once

#include "sdfg/data_flow/library_nodes/math/tensor/tensor_node.h"

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/serializer/json_serializer.h"

namespace sdfg {
namespace math {
namespace tensor {

class ReduceNode : public TensorNode {
protected:
    std::vector<symbolic::Expression> shape_;
    std::vector<int64_t> axes_;
    bool keepdims_;

public:
    ReduceNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const data_flow::LibraryNodeCode& code,
        const std::vector<symbolic::Expression>& shape,
        const std::vector<int64_t>& axes,
        bool keepdims
    );

    const std::vector<symbolic::Expression>& shape() const { return shape_; }

    const std::vector<int64_t>& axes() const { return axes_; }

    bool keepdims() const { return keepdims_; }

    symbolic::SymbolSet symbols() const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    bool expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual bool expand_reduction(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::Sequence& body,
        const std::string& input_name,
        const std::string& output_name,
        const types::IType& input_type,
        const types::IType& output_type,
        const data_flow::Subset& input_subset,
        const data_flow::Subset& output_subset
    ) = 0;

    virtual std::string identity() const = 0;
};

template<typename T>
class ReduceNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override {
        const ReduceNode& reduce_node = static_cast<const ReduceNode&>(library_node);
        nlohmann::json j;

        j["code"] = reduce_node.code().value();

        serializer::JSONSerializer serializer;
        j["shape"] = nlohmann::json::array();
        for (auto& dim : reduce_node.shape()) {
            j["shape"].push_back(serializer.expression(dim));
        }

        j["axes"] = reduce_node.axes();
        j["keepdims"] = reduce_node.keepdims();

        return j;
    }

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override {
        // Assertions for required fields
        assert(j.contains("element_id"));
        assert(j.contains("code"));
        assert(j.contains("debug_info"));
        assert(j.contains("shape"));
        assert(j.contains("axes"));
        assert(j.contains("keepdims"));

        auto code = j["code"].get<std::string>();

        std::vector<symbolic::Expression> shape;
        for (const auto& dim : j["shape"]) {
            shape.push_back(symbolic::parse(dim.get<std::string>()));
        }

        std::vector<int64_t> axes = j["axes"].get<std::vector<int64_t>>();
        bool keepdims = j["keepdims"].get<bool>();

        // Extract debug info using JSONSerializer
        sdfg::serializer::JSONSerializer serializer;
        DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

        return static_cast<ReduceNode&>(builder.add_library_node<T>(parent, debug_info, shape, axes, keepdims));
    }
};

} // namespace tensor
} // namespace math
} // namespace sdfg
