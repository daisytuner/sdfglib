#pragma once

#include "sdfg/data_flow/library_node.h"

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/serializer/json_serializer.h"

namespace sdfg {
namespace data_flow {

inline LibraryNodeCode LibraryNodeType_Call("Call");

class CallNode : public LibraryNode {
private:
    std::string function_name_;

public:
    CallNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        DataFlowGraph& parent,
        const std::string& function_name,
        const std::vector<std::string>& outputs,
        const std::vector<std::string>& inputs
    );

    const std::string& function_name() const;

    const types::Function& function_type(const Function& sdfg) const;

    bool is_void(const Function& sdfg) const;

    void validate(const Function& function) const override;

    symbolic::SymbolSet symbols() const override;

    void replace(const symbolic::Expression& old_expression, const symbolic::Expression& new_expression) override;

    std::unique_ptr<DataFlowNode> clone(size_t element_id, const graph::Vertex vertex, DataFlowGraph& parent)
        const override;
};

class CallNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const LibraryNode& library_node) override;

    LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

class CallNodeDispatcher : public codegen::LibraryNodeDispatcher {
public:
    CallNodeDispatcher(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const DataFlowGraph& data_flow_graph,
        const CallNode& node
    );

    void dispatch_code(
        codegen::PrettyPrinter& stream,
        codegen::PrettyPrinter& globals_stream,
        codegen::CodeSnippetFactory& library_snippet_factory
    ) override;
};

} // namespace data_flow
} // namespace sdfg
