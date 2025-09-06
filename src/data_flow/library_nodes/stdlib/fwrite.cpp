#include "sdfg/data_flow/library_nodes/stdlib/fwrite.h"

namespace sdfg {
namespace stdlib {

FWriteNode::FWriteNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const symbolic::Expression& size,
    const symbolic::Expression& count
)
    : LibraryNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_FWrite,
          {"_ret", "_stream"},
          {"_buffer", "_stream"},
          true,
          data_flow::ImplementationType_NONE
      ),
      size_(size), count_(count) {}

const symbolic::Expression& FWriteNode::size() const { return size_; }

const symbolic::Expression& FWriteNode::count() const { return count_; }

void FWriteNode::validate(const Function& function) const {}

symbolic::SymbolSet FWriteNode::symbols() const {
    symbolic::SymbolSet syms = symbolic::atoms(this->count_);
    for (auto& sym : symbolic::atoms(this->size_)) {
        syms.insert(sym);
    }
    return syms;
}

std::unique_ptr<data_flow::DataFlowNode> FWriteNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::make_unique<FWriteNode>(element_id, debug_info_, vertex, parent, size_, count_);
}

void FWriteNode::replace(const symbolic::Expression& old_expression, const symbolic::Expression& new_expression) {
    this->size_ = symbolic::subs(this->size_, old_expression, new_expression);
    this->count_ = symbolic::subs(this->count_, old_expression, new_expression);
}

nlohmann::json FWriteNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const FWriteNode& node = static_cast<const FWriteNode&>(library_node);

    nlohmann::json j;
    j["code"] = node.code().value();

    sdfg::serializer::JSONSerializer serializer;
    j["size"] = serializer.expression(node.size());
    j["count"] = serializer.expression(node.count());

    return j;
}

data_flow::LibraryNode& FWriteNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    assert(j.contains("code"));
    assert(j.contains("debug_info"));
    assert(j.contains("size"));
    assert(j.contains("count"));

    auto code = j["code"].get<std::string>();
    if (code != LibraryNodeType_FWrite.value()) {
        throw InvalidSDFGException("Invalid library node code");
    }

    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    SymEngine::Expression size(j.at("size"));
    SymEngine::Expression count(j.at("count"));

    return builder.add_library_node<FWriteNode>(parent, debug_info, size, count);
}

FWriteNodeDispatcher::FWriteNodeDispatcher(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const FWriteNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void FWriteNodeDispatcher::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    auto& fwrite_node = static_cast<const FWriteNode&>(node_);

    stream << fwrite_node.outputs().at(0);
    stream << " = ";
    stream << "fwrite(";
    stream << fwrite_node.inputs().at(0) << ", ";
    stream << language_extension_.expression(fwrite_node.size()) << ", ";
    stream << language_extension_.expression(fwrite_node.count()) << ", ";
    stream << fwrite_node.inputs().at(1) << ");";
    stream << std::endl;
}

} // namespace stdlib
} // namespace sdfg
