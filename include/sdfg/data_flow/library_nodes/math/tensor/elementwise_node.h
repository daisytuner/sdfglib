#pragma once

#include "sdfg/data_flow/library_nodes/math/math_node.h"

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/serializer/json_serializer.h"

namespace sdfg {
namespace math {
namespace tensor {

class ElementWiseUnaryNode : public math::MathNode {
protected:
    std::vector<symbolic::Expression> shape_;

public:
    ElementWiseUnaryNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const data_flow::LibraryNodeCode& code,
        const std::vector<symbolic::Expression>& shape
    );

    const std::vector<symbolic::Expression>& shape() const { return shape_; }

    symbolic::SymbolSet symbols() const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    void validate(const Function& function) const override;

    bool expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual bool expand_operation(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::Sequence& body,
        const std::string& input_name,
        const std::string& output_name,
        const types::IType& input_type,
        const types::IType& output_type,
        const data_flow::Subset& subset
    ) = 0;
};

template<typename T>
class ElementWiseUnaryNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override {
        const ElementWiseUnaryNode& elem_node = static_cast<const ElementWiseUnaryNode&>(library_node);
        nlohmann::json j;

        j["code"] = elem_node.code().value();

        serializer::JSONSerializer serializer;
        j["shape"] = nlohmann::json::array();
        for (auto& dim : elem_node.shape()) {
            j["shape"].push_back(serializer.expression(dim));
        }

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

        auto code = j["code"].get<std::string>();

        std::vector<symbolic::Expression> shape;
        for (const auto& dim : j["shape"]) {
            shape.push_back(symbolic::parse(dim.get<std::string>()));
        }

        // Extract debug info using JSONSerializer
        sdfg::serializer::JSONSerializer serializer;
        DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

        return static_cast<ElementWiseUnaryNode&>(builder.add_library_node<T>(parent, debug_info, shape));
    }
};

class ElementWiseBinaryNode : public math::MathNode {
protected:
    std::vector<symbolic::Expression> shape_;

public:
    ElementWiseBinaryNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const data_flow::LibraryNodeCode& code,
        const std::vector<symbolic::Expression>& shape
    );

    const std::vector<symbolic::Expression>& shape() const { return shape_; }

    symbolic::SymbolSet symbols() const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    void validate(const Function& function) const override;

    bool expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual bool expand_operation(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::Sequence& body,
        const std::string& input_name_a,
        const std::string& input_name_b,
        const std::string& output_name,
        const types::IType& input_type_a,
        const types::IType& input_type_b,
        const types::IType& output_type,
        const data_flow::Subset& subset
    ) = 0;
};

template<typename T>
class ElementWiseBinaryNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override {
        const ElementWiseBinaryNode& elem_node = static_cast<const ElementWiseBinaryNode&>(library_node);
        nlohmann::json j;

        j["code"] = elem_node.code().value();

        serializer::JSONSerializer serializer;
        j["shape"] = nlohmann::json::array();
        for (auto& dim : elem_node.shape()) {
            j["shape"].push_back(serializer.expression(dim));
        }

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

        auto code = j["code"].get<std::string>();

        std::vector<symbolic::Expression> shape;
        for (const auto& dim : j["shape"]) {
            shape.push_back(symbolic::parse(dim.get<std::string>()));
        }

        // Extract debug info using JSONSerializer
        sdfg::serializer::JSONSerializer serializer;
        DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

        return static_cast<ElementWiseBinaryNode&>(builder.add_library_node<T>(parent, debug_info, shape));
    }
};

} // namespace tensor
} // namespace math
} // namespace sdfg
