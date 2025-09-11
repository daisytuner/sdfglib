#include "sdfg/data_flow/library_nodes/stdlib/malloc.h"

namespace sdfg {
namespace stdlib {

MallocNode::MallocNode(
    size_t element_id,
    const DebugInfoRegion& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const symbolic::Expression& size
)
    : LibraryNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_Malloc,
          {"_ret"},
          {},
          true,
          data_flow::ImplementationType_NONE
      ),
      size_(size) {}

const symbolic::Expression& MallocNode::size() const { return size_; }

void MallocNode::validate(const Function& function) const {}

symbolic::SymbolSet MallocNode::symbols() const { return symbolic::atoms(this->size_); }

std::unique_ptr<data_flow::DataFlowNode> MallocNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::make_unique<MallocNode>(element_id, debug_info_, vertex, parent, size_);
}

void MallocNode::replace(const symbolic::Expression& old_expression, const symbolic::Expression& new_expression) {
    this->size_ = symbolic::subs(this->size_, old_expression, new_expression);
}

nlohmann::json MallocNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const MallocNode& node = static_cast<const MallocNode&>(library_node);

    nlohmann::json j;
    j["code"] = node.code().value();

    sdfg::serializer::JSONSerializer serializer;
    j["size"] = serializer.expression(node.size());

    return j;
}

data_flow::LibraryNode& MallocNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    assert(j.contains("code"));
    assert(j.contains("debug_info"));
    assert(j.contains("size"));

    auto code = j["code"].get<std::string>();
    if (code != LibraryNodeType_Malloc.value()) {
        throw InvalidSDFGException("Invalid library node code");
    }

    sdfg::serializer::JSONSerializer serializer;
    DebugInfoRegion debug_info = serializer.json_to_debug_info_region(j["debug_info"], builder.debug_info());

    SymEngine::Expression size(j.at("size"));

    return builder.add_library_node<MallocNode>(parent, debug_info, size);
}

MallocNodeDispatcher::MallocNodeDispatcher(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const MallocNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void MallocNodeDispatcher::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    auto& malloc_node = static_cast<const MallocNode&>(node_);

    stream << malloc_node.outputs().at(0);
    stream << " = ";
    stream << "malloc(" << language_extension_.expression(malloc_node.size()) << ")" << ";";
    stream << std::endl;
}

} // namespace stdlib
} // namespace sdfg
