#include "sdfg/data_flow/library_nodes/stdlib/fprintf.h"

namespace sdfg {
namespace stdlib {

FprintfNode::FprintfNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<std::string>& inputs
)
    : LibraryNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_Fprintf,
          {"_out"},
          inputs,
          true,
          data_flow::ImplementationType_NONE
      ) {}

void FprintfNode::validate(const Function& function) const {}

symbolic::SymbolSet FprintfNode::symbols() const { return {}; }

std::unique_ptr<data_flow::DataFlowNode> FprintfNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::make_unique<FprintfNode>(element_id, debug_info_, vertex, parent, this->inputs_);
}

void FprintfNode::replace(const symbolic::Expression& old_expression, const symbolic::Expression& new_expression) {}

nlohmann::json FprintfNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const FprintfNode& node = static_cast<const FprintfNode&>(library_node);

    nlohmann::json j;
    j["code"] = node.code().value();
    j["outputs"] = node.outputs();
    j["inputs"] = node.inputs();
    j["side_effect"] = node.side_effect();

    return j;
}

data_flow::LibraryNode& FprintfNodeSerializer::deserialize(
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
    if (code != LibraryNodeType_Fprintf.value()) {
        throw std::runtime_error("Invalid library node code");
    }

    // Extract debug info using JSONSerializer
    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    std::vector<std::string> inputs = j["inputs"].get<std::vector<std::string>>();

    return builder.add_library_node<FprintfNode>(parent, debug_info, inputs);
}

FprintfNodeDispatcher::FprintfNodeDispatcher(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const FprintfNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void FprintfNodeDispatcher::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    auto& fprintf_node = static_cast<const FprintfNode&>(node_);

    stream << fprintf_node.outputs().at(0);
    stream << " = ";
    stream << "fprintf(";
    for (size_t i = 0; i < fprintf_node.inputs().size(); ++i) {
        stream << ", " << fprintf_node.inputs().at(i);
    }
    stream << ");";
    stream << std::endl;
}

} // namespace stdlib
} // namespace sdfg
