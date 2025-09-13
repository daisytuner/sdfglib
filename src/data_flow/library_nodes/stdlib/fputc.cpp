#include "sdfg/data_flow/library_nodes/stdlib/fputc.h"

namespace sdfg {
namespace stdlib {

FputcNode::FputcNode(
    size_t element_id, const DebugInfo& debug_info, const graph::Vertex vertex, data_flow::DataFlowGraph& parent
)
    : LibraryNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_Fputc,
          {"_ret", "_stream"},
          {"_character", "_stream"},
          true,
          data_flow::ImplementationType_NONE
      ) {}

void FputcNode::validate(const Function& function) const {}

symbolic::SymbolSet FputcNode::symbols() const { return {}; }

std::unique_ptr<data_flow::DataFlowNode> FputcNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::make_unique<FputcNode>(element_id, debug_info_, vertex, parent);
}

void FputcNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {}

nlohmann::json FputcNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const FputcNode& node = static_cast<const FputcNode&>(library_node);

    nlohmann::json j;
    j["code"] = node.code().value();

    return j;
}

data_flow::LibraryNode& FputcNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    assert(j.contains("code"));
    assert(j.contains("debug_info"));

    auto code = j["code"].get<std::string>();
    if (code != LibraryNodeType_Fputc.value()) {
        throw InvalidSDFGException("Invalid library node code");
    }

    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    return builder.add_library_node<FputcNode>(parent, debug_info);
}

FputcNodeDispatcher::FputcNodeDispatcher(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const FputcNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void FputcNodeDispatcher::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    auto& fputc_node = static_cast<const FputcNode&>(node_);

    stream << fputc_node.outputs().at(0);
    stream << " = ";
    stream << "fputc(";
    stream << fputc_node.inputs().at(0) << ", ";
    stream << fputc_node.inputs().at(1) << ");";
    stream << std::endl;
}

} // namespace stdlib
} // namespace sdfg
