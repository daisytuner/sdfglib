#include "sdfg/data_flow/library_nodes/math/intrinsic.h"

namespace sdfg {
namespace math {

IntrinsicNode::IntrinsicNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::string& name,
    size_t arity
)
    : MathNode(element_id, debug_info, vertex, parent, LibraryNodeType_Intrinsic, {"_out"}, {}, data_flow::ImplementationType_NONE),
      name_(name) {
        for (size_t i = 0; i < arity; i++) {
            this->inputs_.push_back("_in" + std::to_string(i + 1));
        }
      }

const std::string& IntrinsicNode::name() const {
    return this->name_;
}

void IntrinsicNode::validate(const Function& function) const {}

std::unique_ptr<data_flow::DataFlowNode> IntrinsicNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<IntrinsicNode>(new IntrinsicNode(
        element_id,
        this->debug_info(),
        vertex,
        parent,
        this->name_,
        this->inputs_.size()
    ));
}

nlohmann::json IntrinsicNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const IntrinsicNode& node = static_cast<const IntrinsicNode&>(library_node);
    nlohmann::json j;

    serializer::JSONSerializer serializer;
    j["code"] = node.code().value();
    j["name"] = node.name();
    j["arity"] = node.inputs().size();

    return j;
}

data_flow::LibraryNode& IntrinsicNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    // Assertions for required fields
    assert(j.contains("element_id"));
    assert(j.contains("code"));
    assert(j.contains("debug_info"));

    auto code = j["code"].get<std::string>();
    if (code != LibraryNodeType_Intrinsic.value()) {
        throw std::runtime_error("Invalid library node code");
    }

    // Extract debug info using JSONSerializer
    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    auto name = j["name"].get<std::string>();
    auto arity = j["arity"].get<size_t>();

    return builder.add_library_node<IntrinsicNode>(parent, debug_info, name, arity);
}

IntrinsicNodeDispatcher::IntrinsicNodeDispatcher(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const IntrinsicNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void IntrinsicNodeDispatcher::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    stream << "{" << std::endl;
    stream.setIndent(stream.indent() + 4);

    auto& node = static_cast<const IntrinsicNode&>(this->node_);

    stream << node.outputs().at(0) << " = ";
    stream << node.name() << "(";
    for (size_t i = 0; i < node.inputs().size(); i++) {
        stream << node.inputs().at(i);
        if (i < node.inputs().size() - 1) {
            stream << ", ";
        }
    }
    stream << ");" << std::endl;

    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
}

} // namespace math
} // namespace sdfg
