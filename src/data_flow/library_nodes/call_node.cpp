#include "sdfg/data_flow/library_nodes/call_node.h"

namespace sdfg {
namespace data_flow {

CallNode::CallNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::string& function_name,
    const std::vector<std::string>& outputs,
    const std::vector<std::string>& inputs
)
    : LibraryNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_Call,
          outputs,
          inputs,
          true,
          data_flow::ImplementationType_NONE
      ),
      function_name_(function_name) {}

const std::string& CallNode::function_name() const { return this->function_name_; }

const types::Function& CallNode::function_type(const Function& sdfg) const {
    // function_name is a symbol referring to a global variable of type Function
    return dynamic_cast<const types::Function&>(sdfg.type(this->function_name_));
}

bool CallNode::is_void(const Function& sdfg) const {
    auto& func_type = this->function_type(sdfg);
    if (func_type.return_type().type_id() == types::TypeID::Scalar) {
        auto& ret_type = static_cast<const types::Scalar&>(func_type.return_type());
        if (ret_type.primitive_type() == types::PrimitiveType::Void) {
            return true;
        }
    }
    return false;
}

void CallNode::validate(const Function& function) const {
    if (!function.exists(this->function_name_)) {
        throw InvalidSDFGException("CallNode: Function '" + this->function_name_ + "' does not exist.");
    }
    auto& type = function.type(this->function_name_);
    if (!dynamic_cast<const types::Function*>(&type)) {
        throw InvalidSDFGException("CallNode: '" + this->function_name_ + "' is not a function.");
    }
    auto& func_type = static_cast<const types::Function&>(type);

    if (inputs_.size() != func_type.num_params()) {
        throw InvalidSDFGException(
            "CallNode: Number of inputs does not match number of function parameters. Expected " +
            std::to_string(func_type.num_params()) + ", got " + std::to_string(inputs_.size())
        );
    }
    if (!this->is_void(function) && outputs_.size() < 1) {
        throw InvalidSDFGException(
            "CallNode: Non-void function must have at least one output to store the return value."
        );
    }
}

symbolic::SymbolSet CallNode::symbols() const { return {}; }

std::unique_ptr<data_flow::DataFlowNode> CallNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::make_unique<CallNode>(element_id, debug_info_, vertex, parent, function_name_, outputs_, inputs_);
}

void CallNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {}

nlohmann::json CallNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const CallNode& node = static_cast<const CallNode&>(library_node);

    nlohmann::json j;
    j["code"] = node.code().value();
    j["function_name"] = node.function_name();
    j["outputs"] = node.outputs();
    j["inputs"] = node.inputs();

    return j;
}

data_flow::LibraryNode& CallNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    assert(j.contains("code"));
    assert(j.contains("function_name"));
    assert(j.contains("outputs"));
    assert(j.contains("inputs"));
    assert(j.contains("debug_info"));

    auto code = j["code"].get<std::string>();
    if (code != LibraryNodeType_Call.value()) {
        throw InvalidSDFGException("Invalid library node code");
    }

    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    std::string function_name = j["function_name"].get<std::string>();
    auto outputs = j["outputs"].get<std::vector<std::string>>();
    auto inputs = j["inputs"].get<std::vector<std::string>>();

    return builder.add_library_node<CallNode>(parent, debug_info, function_name, outputs, inputs);
}

CallNodeDispatcher::CallNodeDispatcher(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const CallNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void CallNodeDispatcher::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    auto& node = static_cast<const CallNode&>(node_);

    // Declare function
    if (this->language_extension_.language() == "C") {
        globals_stream << "extern ";
    } else if (this->language_extension_.language() == "C++") {
        globals_stream << "extern \"C\" ";
    }
    globals_stream << language_extension_.declaration(node.function_name(), node.function_type(this->function_)) << ";"
                   << std::endl;

    if (!node.is_void(function_)) {
        stream << node.outputs().at(0) << " = ";
    }
    stream << node.function_name() << "(";
    for (size_t i = 0; i < node.inputs().size(); ++i) {
        stream << node.inputs().at(i);
        if (i < node.inputs().size() - 1) {
            stream << ", ";
        }
    }
    stream << ")" << ";";
    stream << std::endl;
}

} // namespace data_flow
} // namespace sdfg
