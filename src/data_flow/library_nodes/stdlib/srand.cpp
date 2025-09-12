#include "sdfg/data_flow/library_nodes/stdlib/srand.h"

namespace sdfg {
namespace stdlib {

SrandNode::SrandNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const symbolic::Expression& seed
)
    : LibraryNode(
          element_id, debug_info, vertex, parent, LibraryNodeType_Srand, {}, {}, true, data_flow::ImplementationType_NONE
      ),
      seed_(seed) {}

const symbolic::Expression& SrandNode::seed() const { return seed_; }

void SrandNode::validate(const Function& function) const {}

symbolic::SymbolSet SrandNode::symbols() const { return symbolic::atoms(this->seed_); }

std::unique_ptr<data_flow::DataFlowNode> SrandNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::make_unique<SrandNode>(element_id, debug_info_, vertex, parent, seed_);
}

void SrandNode::replace(const symbolic::Expression& old_expression, const symbolic::Expression& new_expression) {
    this->seed_ = symbolic::subs(this->seed_, old_expression, new_expression);
}

nlohmann::json SrandNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const SrandNode& node = static_cast<const SrandNode&>(library_node);

    nlohmann::json j;
    j["code"] = node.code().value();

    sdfg::serializer::JSONSerializer serializer;
    j["seed"] = serializer.expression(node.seed());

    return j;
}

data_flow::LibraryNode& SrandNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    // Assertions for required fields
    assert(j.contains("code"));
    assert(j.contains("debug_info"));
    assert(j.contains("seed"));

    auto code = j["code"].get<std::string>();
    if (code != LibraryNodeType_Srand.value()) {
        throw InvalidSDFGException("Invalid library node code");
    }

    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    auto seed = symbolic::parse(j.at("seed"));

    return builder.add_library_node<SrandNode>(parent, debug_info, seed);
}

SrandNodeDispatcher::SrandNodeDispatcher(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const SrandNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void SrandNodeDispatcher::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    auto& srand_node = static_cast<const SrandNode&>(node_);

    stream << "srand(" << language_extension_.expression(srand_node.seed()) << ")" << ";" << std::endl;
}

} // namespace stdlib
} // namespace sdfg
