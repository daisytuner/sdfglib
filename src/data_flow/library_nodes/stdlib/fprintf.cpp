#include "sdfg/data_flow/library_nodes/stdlib/fprintf.h"

namespace sdfg {
namespace stdlib {

FprintfNode::FprintfNode(
    size_t element_id,
    const DebugInfoRegion& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<std::string>& args
)
    : LibraryNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_Fprintf,
          {"_ret", "_stream"},
          {"_stream", "_format"},
          true,
          data_flow::ImplementationType_NONE
      ),
      args_(args) {
    for (auto& arg : args) {
        inputs_.push_back(arg);
    }
}

void FprintfNode::validate(const Function& function) const {}

symbolic::SymbolSet FprintfNode::symbols() const { return {}; }

std::unique_ptr<data_flow::DataFlowNode> FprintfNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::make_unique<FprintfNode>(element_id, debug_info_, vertex, parent, this->args_);
}

void FprintfNode::replace(const symbolic::Expression& old_expression, const symbolic::Expression& new_expression) {}

nlohmann::json FprintfNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const FprintfNode& node = static_cast<const FprintfNode&>(library_node);

    nlohmann::json j;
    j["code"] = node.code().value();
    j["args"] = node.args();

    return j;
}

data_flow::LibraryNode& FprintfNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    assert(j.contains("code"));
    assert(j.contains("debug_info"));
    assert(j.contains("args"));

    auto code = j["code"].get<std::string>();
    if (code != LibraryNodeType_Fprintf.value()) {
        throw InvalidSDFGException("Invalid library node code");
    }

    sdfg::serializer::JSONSerializer serializer;
    DebugInfoRegion debug_info = serializer.json_to_debug_info_region(j["debug_info"], builder.debug_info());

    std::vector<std::string> args = j["args"].get<std::vector<std::string>>();

    return builder.add_library_node<FprintfNode>(parent, debug_info, args);
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
        stream << fprintf_node.inputs().at(i);
        if (i < fprintf_node.inputs().size() - 1) {
            stream << ", ";
        }
    }
    stream << ");";
    stream << std::endl;
}

} // namespace stdlib
} // namespace sdfg
