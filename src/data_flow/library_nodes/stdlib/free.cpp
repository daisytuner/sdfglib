#include "sdfg/data_flow/library_nodes/stdlib/free.h"

namespace sdfg {
namespace stdlib {

FreeNode::FreeNode(
    size_t element_id, const DebugInfo& debug_info, const graph::Vertex vertex, data_flow::DataFlowGraph& parent
)
    : LibraryNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_Free,
          {},
          {"_in"},
          true,
          data_flow::ImplementationType_NONE
      ) {}


void FreeNode::validate(const Function& function) const {}

symbolic::SymbolSet FreeNode::symbols() const { return {}; }

std::unique_ptr<data_flow::DataFlowNode> FreeNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::make_unique<FreeNode>(element_id, debug_info_, vertex, parent);
}

void FreeNode::replace(const symbolic::Expression& old_expression, const symbolic::Expression& new_expression) {
    // Do nothing
    return;
}

nlohmann::json FreeNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const FreeNode& node = static_cast<const FreeNode&>(library_node);

    nlohmann::json j;
    j["code"] = node.code().value();
    j["outputs"] = node.outputs();
    j["inputs"] = node.inputs();
    j["side_effect"] = node.side_effect();

    return j;
}

data_flow::LibraryNode& FreeNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    // Assertions for required fields
    assert(j.contains("element_id"));
    assert(j.contains("code"));
    assert(j.contains("outputs"));
    assert(j.contains("inputs"));
    assert(j.contains("debug_info"));

    auto code = j["code"].get<std::string>();
    if (code != LibraryNodeType_Free.value()) {
        throw std::runtime_error("Invalid library node code");
    }

    // Extract debug info using JSONSerializer
    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    return builder.add_library_node<FreeNode>(parent, debug_info);
}

FreeNodeDispatcher::FreeNodeDispatcher(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const FreeNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void FreeNodeDispatcher::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    auto& free_node = static_cast<const FreeNode&>(node_);

    stream << "free(" << free_node.inputs().at(0) << ");" << std::endl;
}

} // namespace stdlib
} // namespace sdfg
