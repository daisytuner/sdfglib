#pragma once

#include "sdfg/data_flow/library_nodes/math/math_node.h"

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/serializer/json_serializer.h"

namespace sdfg {
namespace math {
namespace ml {

class ElementWiseUnaryNode : public math::MathNode {
protected:
    std::unordered_map<std::string, std::string> attributes_;

public:
    ElementWiseUnaryNode(
        size_t element_id,
        const DebugInfoRegion& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const data_flow::LibraryNodeCode& code,
        const std::unordered_map<std::string, std::string>& attributes
    );

    void set_attributes(const std::unordered_map<std::string, std::string>& attributes) { attributes_ = attributes; }

    const std::unordered_map<std::string, std::string>& attributes() const { return attributes_; }

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
        j["attributes"] = elem_node.attributes();

        return j;
    }

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override {
        // Assertions for required fields
        assert(j.contains("element_id"));
        assert(j.contains("code"));
        assert(j.contains("debug_info"));
        assert(j.contains("attributes"));

        auto code = j["code"].get<std::string>();
        auto attributes = j["attributes"].get<std::unordered_map<std::string, std::string>>();

        // Extract debug info using JSONSerializer
        sdfg::serializer::JSONSerializer serializer;
        DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

        auto& node = static_cast<ElementWiseUnaryNode&>(builder.add_library_node<T>(parent, debug_info));
        node.set_attributes(attributes);

        return node;
    }
};

class ElementWiseBinaryNode : public math::MathNode {
protected:
    std::unordered_map<std::string, std::string> attributes_;

public:
    ElementWiseBinaryNode(
        size_t element_id,
        const DebugInfoRegion& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const data_flow::LibraryNodeCode& code,
        const std::unordered_map<std::string, std::string>& attributes
    );

    void set_attributes(const std::unordered_map<std::string, std::string>& attributes) { attributes_ = attributes; }

    const std::unordered_map<std::string, std::string>& attributes() const { return attributes_; }

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
        j["attributes"] = elem_node.attributes();

        return j;
    }

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override {
        // Assertions for required fields
        assert(j.contains("element_id"));
        assert(j.contains("code"));
        assert(j.contains("debug_info"));
        assert(j.contains("attributes"));

        auto code = j["code"].get<std::string>();
        auto attributes = j["attributes"].get<std::unordered_map<std::string, std::string>>();

        // Extract debug info using JSONSerializer
        sdfg::serializer::JSONSerializer serializer;
        DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

        auto& node = static_cast<ElementWiseBinaryNode&>(builder.add_library_node<T>(parent, debug_info));
        node.set_attributes(attributes);

        return node;
    }
};

} // namespace ml
} // namespace math
} // namespace sdfg
