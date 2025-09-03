#include "sdfg/data_flow/library_nodes/stdlib/fputc.h"

namespace sdfg {
namespace stdlib {

FPutcNode::FPutcNode(
    size_t element_id, const DebugInfo& debug_info, const graph::Vertex vertex, data_flow::DataFlowGraph& parent
)
    : LibraryNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_FPutc,
          {"_out"},
          {"character", "stream"},
          true,
          data_flow::ImplementationType_NONE
      ) {}

void FPutcNode::validate(const Function& function) const {}

symbolic::SymbolSet FPutcNode::symbols() const { return {}; }

std::unique_ptr<data_flow::DataFlowNode> FPutcNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::make_unique<FPutcNode>(element_id, debug_info_, vertex, parent);
}

void FPutcNode::replace(const symbolic::Expression& old_expression, const symbolic::Expression& new_expression) {}

nlohmann::json FPutcNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const FPutcNode& node = static_cast<const FPutcNode&>(library_node);

    nlohmann::json j;
    j["code"] = node.code().value();
    j["outputs"] = node.outputs();
    j["inputs"] = node.inputs();
    j["side_effect"] = node.side_effect();

    return j;
}

data_flow::LibraryNode& FPutcNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    // Assertions for required fields
    assert(j.contains("element_id"));
    assert(j.contains("code"));
    assert(j.contains("outputs"));
    assert(j.contains("inputs"));
    assert(j.contains("debug_info"));
    assert(j.contains("size"));

    auto code = j["code"].get<std::string>();
    if (code != LibraryNodeType_FPutc.value()) {
        throw std::runtime_error("Invalid library node code");
    }

    // Extract debug info using JSONSerializer
    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    return builder.add_library_node<FPutcNode>(parent, debug_info);
}

FPutcNodeDispatcher::FPutcNodeDispatcher(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const FPutcNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void FPutcNodeDispatcher::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    auto& fputc_node = static_cast<const FPutcNode&>(node_);

    stream << fputc_node.outputs().at(0);
    stream << " = ";
    stream << "fputc(";
    stream << fputc_node.inputs().at(0) << ", ";
    stream << fputc_node.inputs().at(1) << ");";
    stream << std::endl;
}

} // namespace stdlib
} // namespace sdfg
