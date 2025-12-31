#pragma once

#include "sdfg/data_flow/library_nodes/math/math_node.h"

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace math {
namespace cmath {

inline data_flow::LibraryNodeCode LibraryNodeType_CMath("CMath");
inline data_flow::LibraryNodeCode LibraryNodeType_CMath_Deprecated("Intrinsic");

class CMathNode : public math::MathNode {
private:
    std::string name_;

public:
    CMathNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const std::string& name,
        size_t arity
    );

    const std::string& name() const;

    void validate(const Function& function) const override;

    symbolic::SymbolSet symbols() const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    bool expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override {
        return false;
    };

    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;

    virtual symbolic::Expression flop() const override;
};

class CMathNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

class CMathNodeDispatcher : public codegen::LibraryNodeDispatcher {
public:
    CMathNodeDispatcher(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const CMathNode& node
    );

    void dispatch_code(
        codegen::PrettyPrinter& stream,
        codegen::PrettyPrinter& globals_stream,
        codegen::CodeSnippetFactory& library_snippet_factory
    ) override;
};

} // namespace cmath
} // namespace math
} // namespace sdfg
