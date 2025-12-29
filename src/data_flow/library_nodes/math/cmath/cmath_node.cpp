#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace math {
namespace cmath {

CMathNode::CMathNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::string& name,
    size_t arity
)
    : MathNode(
          element_id, debug_info, vertex, parent, LibraryNodeType_CMath, {"_out"}, {}, data_flow::ImplementationType_NONE
      ),
      name_(name) {
    for (size_t i = 0; i < arity; i++) {
        this->inputs_.push_back("_in" + std::to_string(i + 1));
    }
}

const std::string& CMathNode::name() const { return this->name_; }

symbolic::SymbolSet CMathNode::symbols() const { return {}; }

void CMathNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    return;
}


void CMathNode::validate(const Function& function) const {}

std::unique_ptr<data_flow::DataFlowNode> CMathNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<
        CMathNode>(new CMathNode(element_id, this->debug_info(), vertex, parent, this->name_, this->inputs_.size()));
}

symbolic::Expression CMathNode::flop() const { return symbolic::one(); }

nlohmann::json CMathNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const CMathNode& node = static_cast<const CMathNode&>(library_node);
    nlohmann::json j;

    serializer::JSONSerializer serializer;
    j["code"] = node.code().value();
    j["name"] = node.name();
    j["arity"] = node.inputs().size();

    return j;
}

data_flow::LibraryNode& CMathNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    // Assertions for required fields
    assert(j.contains("element_id"));
    assert(j.contains("code"));
    assert(j.contains("debug_info"));

    auto code = j["code"].get<std::string>();
    // Backward compatibility
    if (code != LibraryNodeType_CMath.value() && code != LibraryNodeType_CMath_Deprecated.value()) {
        throw std::runtime_error("Invalid library node code");
    }

    // Extract debug info using JSONSerializer
    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    auto name = j["name"].get<std::string>();
    auto arity = j["arity"].get<size_t>();

    return builder.add_library_node<CMathNode>(parent, debug_info, name, arity);
}

CMathNodeDispatcher::CMathNodeDispatcher(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const CMathNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void CMathNodeDispatcher::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    stream << "{" << std::endl;
    stream.setIndent(stream.indent() + 4);

    auto& node = static_cast<const CMathNode&>(this->node_);

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

} // namespace cmath
} // namespace math
} // namespace sdfg
