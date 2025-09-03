#include "sdfg/data_flow/library_nodes/stdlib/rand.h"

namespace sdfg {
namespace stdlib {

RandNode::RandNode(
    size_t element_id, const DebugInfo& debug_info, const graph::Vertex vertex, data_flow::DataFlowGraph& parent
)
    : LibraryNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_Rand,
          {"_out"},
          {},
          true,
          data_flow::ImplementationType_NONE
      ) {}

void RandNode::validate(const Function& function) const {}

symbolic::SymbolSet RandNode::symbols() const { return {}; }

std::unique_ptr<data_flow::DataFlowNode> RandNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::make_unique<RandNode>(element_id, debug_info_, vertex, parent);
}

void RandNode::replace(const symbolic::Expression& old_expression, const symbolic::Expression& new_expression) {}

nlohmann::json RandNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const RandNode& node = static_cast<const RandNode&>(library_node);

    nlohmann::json j;
    j["code"] = node.code().value();
    j["outputs"] = node.outputs();
    j["inputs"] = node.inputs();
    j["side_effect"] = node.side_effect();

    return j;
}

data_flow::LibraryNode& RandNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    // Assertions for required fields
    assert(j.contains("element_id"));
    assert(j.contains("code"));
    assert(j.contains("outputs"));
    assert(j.contains("inputs"));
    assert(j.contains("debug_info"));

    auto code = j["code"].get<std::string>();
    if (code != LibraryNodeType_Rand.value()) {
        throw std::runtime_error("Invalid library node code");
    }

    // Extract debug info using JSONSerializer
    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    return builder.add_library_node<RandNode>(parent, debug_info);
}

RandNodeDispatcher::RandNodeDispatcher(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const RandNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void RandNodeDispatcher::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    auto& rand_node = static_cast<const RandNode&>(node_);

    stream << rand_node.outputs().at(0) << " = ";
    stream << "rand()" << ";" << std::endl;
}

} // namespace stdlib
} // namespace sdfg
