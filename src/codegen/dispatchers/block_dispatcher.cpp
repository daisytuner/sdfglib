#include "sdfg/codegen/dispatchers/block_dispatcher.h"

#include "sdfg/codegen/dispatchers/node_dispatcher_registry.h"

namespace sdfg {
namespace codegen {

BlockDispatcher::BlockDispatcher(
    LanguageExtension& language_extension,
    StructuredSDFG& sdfg,
    structured_control_flow::Block& node,
    InstrumentationPlan& instrumentation_plan
)
    : sdfg::codegen::NodeDispatcher(language_extension, sdfg, node, instrumentation_plan), node_(node) {

      };

void BlockDispatcher::dispatch_node(
    PrettyPrinter& main_stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory
) {
    if (node_.dataflow().nodes().empty()) {
        return;
    }

    DataFlowDispatcher dispatcher(this->language_extension_, sdfg_, node_.dataflow());
    dispatcher.dispatch(main_stream, globals_stream, library_snippet_factory);
};

DataFlowDispatcher::DataFlowDispatcher(
    LanguageExtension& language_extension, const Function& sdfg, const data_flow::DataFlowGraph& data_flow_graph
)
    : language_extension_(language_extension), function_(sdfg), data_flow_graph_(data_flow_graph) {

      };

void DataFlowDispatcher::
    dispatch(PrettyPrinter& stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory) {
    // Dispatch code nodes in topological order
    auto nodes = this->data_flow_graph_.topological_sort();
    for (auto& node : nodes) {
        if (auto tasklet = dynamic_cast<const data_flow::Tasklet*>(node)) {
            this->dispatch_tasklet(stream, *tasklet);
        } else if (auto libnode = dynamic_cast<const data_flow::LibraryNode*>(node)) {
            this->dispatch_library_node(stream, globals_stream, library_snippet_factory, *libnode);
        } else if (auto access_node = dynamic_cast<const data_flow::AccessNode*>(node)) {
            for (auto& edge : this->data_flow_graph_.out_edges(*access_node)) {
                if (edge.dst_conn() == "ref" && edge.src_conn() == "void") {
                    this->dispatch_ref(stream, edge);
                } else if (edge.src_conn() == "deref" || edge.dst_conn() == "deref") {
                    this->dispatch_deref(stream, edge);
                }
            }
        } else {
            throw InvalidSDFGException("Codegen: Node type not supported");
        }
    }
};

void DataFlowDispatcher::dispatch_ref(PrettyPrinter& stream, const data_flow::Memlet& memlet) {
    stream << "{" << std::endl;
    stream.setIndent(stream.indent() + 4);

    auto& src = dynamic_cast<const data_flow::AccessNode&>(memlet.src());
    auto& src_type = this->function_.type(src.data());
    auto& dst = dynamic_cast<const data_flow::AccessNode&>(memlet.dst());
    auto& dst_type = this->function_.type(dst.data());

    auto& subset = memlet.subset();

    auto& base_type = memlet.base_type();

    std::string src_name = this->language_extension_.access_node(src);
    std::string dst_name = this->language_extension_.access_node(dst);

    stream << dst_name;
    stream << " = ";

    if (symbolic::is_nullptr(symbolic::symbol(src.data())) || helpers::is_number(src.data())) {
        stream << this->language_extension_.expression(symbolic::symbol(src.data()));
    } else if (base_type.type_id() == types::TypeID::Pointer) {
        stream << "&";
        stream << "(" + this->language_extension_.type_cast(src_name, base_type) + ")";
        stream << this->language_extension_.subset(function_, base_type, subset);
    } else {
        stream << "&";
        stream << src_name;
        stream << this->language_extension_.subset(function_, base_type, subset);
    }

    stream << ";";
    stream << std::endl;

    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
};

void DataFlowDispatcher::dispatch_deref(PrettyPrinter& stream, const data_flow::Memlet& memlet) {
    stream << "{" << std::endl;
    stream.setIndent(stream.indent() + 4);

    auto& src = dynamic_cast<const data_flow::AccessNode&>(memlet.src());
    auto& src_type = this->function_.type(src.data());
    auto& dst = dynamic_cast<const data_flow::AccessNode&>(memlet.dst());
    auto& dst_type = this->function_.type(dst.data());

    auto& base_type = static_cast<const types::Pointer&>(memlet.base_type());

    std::string src_name = this->language_extension_.access_node(src);
    std::string dst_name = this->language_extension_.access_node(dst);

    if (memlet.dst_conn() == "void") {
        stream << "(" << this->language_extension_.type_cast(dst_name, base_type) << ")";
        stream << this->language_extension_.subset(function_, base_type, memlet.subset());
    } else {
        stream << dst_name;
    }
    stream << " = ";

    if (symbolic::is_nullptr(symbolic::symbol(src.data()))) {
        stream << this->language_extension_.expression(symbolic::__nullptr__());
        stream << ";" << std::endl;

        stream.setIndent(stream.indent() - 4);
        stream << "}" << std::endl;
        return;
    }

    if (memlet.src_conn() == "void") {
        stream << "(" << this->language_extension_.type_cast(src_name, base_type) << ")";
        stream << this->language_extension_.subset(function_, base_type, memlet.subset());
    } else {
        stream << this->language_extension_.type_cast(src_name, base_type.pointee_type());
    }
    stream << ";" << std::endl;

    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
};

void DataFlowDispatcher::dispatch_tasklet(PrettyPrinter& stream, const data_flow::Tasklet& tasklet) {
    if (tasklet.is_conditional()) {
        stream << "if (" << language_extension_.expression(tasklet.condition()) << ") ";
    }
    stream << "{" << std::endl;
    stream.setIndent(stream.indent() + 4);

    for (auto& iedge : this->data_flow_graph_.in_edges(tasklet)) {
        auto& src = dynamic_cast<const data_flow::AccessNode&>(iedge.src());
        auto& src_type = this->function_.type(src.data());

        std::string src_name = this->language_extension_.access_node(src);

        std::string conn = iedge.dst_conn();
        auto& conn_type = iedge.result_type(this->function_);

        stream << this->language_extension_.declaration(conn, conn_type);
        stream << " = ";

        // Reinterpret cast for opaque pointers
        if (src_type.type_id() == types::TypeID::Pointer) {
            stream << "(" << this->language_extension_.type_cast(src_name, iedge.base_type()) << ")";
        } else {
            stream << src_name;
        }

        stream << this->language_extension_.subset(function_, iedge.base_type(), iedge.subset()) << ";";
        stream << std::endl;
    }

    auto& oedge = *this->data_flow_graph_.out_edges(tasklet).begin();
    std::string out_conn = oedge.src_conn();
    auto& out_conn_type = oedge.result_type(this->function_);

    stream << this->language_extension_.declaration(out_conn, out_conn_type);
    stream << ";" << std::endl;

    stream << std::endl;
    stream << out_conn << " = ";
    stream << this->language_extension_.tasklet(tasklet) << ";" << std::endl;
    stream << std::endl;

    // Write back
    for (auto& oedge : this->data_flow_graph_.out_edges(tasklet)) {
        auto& dst = dynamic_cast<const data_flow::AccessNode&>(oedge.dst());
        auto& dst_type = this->function_.type(dst.data());

        std::string dst_name = this->language_extension_.access_node(dst);

        // Reinterpret cast for opaque pointers
        if (dst_type.type_id() == types::TypeID::Pointer) {
            stream << "(" << this->language_extension_.type_cast(dst_name, oedge.base_type()) << ")";
        } else {
            stream << dst_name;
        }

        stream << this->language_extension_.subset(function_, oedge.base_type(), oedge.subset()) << " = ";
        stream << oedge.src_conn();
        stream << ";" << std::endl;
    }

    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
};

void DataFlowDispatcher::dispatch_library_node(
    PrettyPrinter& stream,
    PrettyPrinter& globals_stream,
    CodeSnippetFactory& library_snippet_factory,
    const data_flow::LibraryNode& libnode
) {
    auto dispatcher_fn =
        LibraryNodeDispatcherRegistry::instance()
            .get_library_node_dispatcher(libnode.code().value() + "::" + libnode.implementation_type().value());
    if (dispatcher_fn) {
        auto dispatcher = dispatcher_fn(this->language_extension_, this->function_, this->data_flow_graph_, libnode);
        dispatcher->dispatch(stream, globals_stream, library_snippet_factory);
    } else {
        throw std::runtime_error(
            "No library node dispatcher found for library node code: " + std::string(libnode.code().value())
        );
    }
};

LibraryNodeDispatcher::LibraryNodeDispatcher(
    LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const data_flow::LibraryNode& node
)
    : language_extension_(language_extension), function_(function), data_flow_graph_(data_flow_graph), node_(node) {};

void LibraryNodeDispatcher::
    dispatch(PrettyPrinter& stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory) {
    stream << "{" << std::endl;
    stream.setIndent(stream.indent() + 4);

    auto& node = node_;
    std::vector<std::string> inputs_by_order;
    std::vector<std::optional<std::string>> outputs_by_order;

    std::unordered_map<std::string, const data_flow::Memlet*> in_edges;
    for (auto& iedge : this->data_flow_graph_.in_edges(node)) {
        in_edges[iedge.dst_conn()] = &iedge;
    }

    for (auto& input : node.inputs()) {
        if (in_edges.find(input) == in_edges.end()) {
            inputs_by_order.push_back(input);
        } else {
            auto& iedge = *in_edges[input];
            auto& src = dynamic_cast<const data_flow::AccessNode&>(iedge.src());
            inputs_by_order.push_back(src.data());
        }
    }

    std::unordered_map<std::string, const data_flow::Memlet*> out_edges;
    for (auto& oedge : this->data_flow_graph_.out_edges(node)) {
        out_edges[oedge.src_conn()] = &oedge;
    }
    for (auto& output : node.outputs()) {
        auto it = out_edges.find(output);
        if (it == out_edges.end()) {
            outputs_by_order.push_back(std::nullopt);
        } else {
            auto& oedge = *it->second;
            auto& dst = dynamic_cast<const data_flow::AccessNode&>(oedge.dst());
            outputs_by_order.push_back(dst.data());
        }
    }

    dispatch_code(stream, globals_stream, library_snippet_factory, inputs_by_order, outputs_by_order);

    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
}


} // namespace codegen
} // namespace sdfg
