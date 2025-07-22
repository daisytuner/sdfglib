#include "sdfg/codegen/dispatchers/block_dispatcher.h"

#include "sdfg/codegen/dispatchers/node_dispatcher_registry.h"

namespace sdfg {
namespace codegen {

BlockDispatcher::BlockDispatcher(
    LanguageExtension& language_extension,
    StructuredSDFG& sdfg,
    structured_control_flow::Block& node,
    Instrumentation& instrumentation
)
    : sdfg::codegen::NodeDispatcher(language_extension, sdfg, node, instrumentation), node_(node) {

      };

void BlockDispatcher::dispatch_node(
    PrettyPrinter& main_stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory
) {
    if (node_.dataflow().nodes().empty()) {
        return;
    }

    DataFlowDispatcher dispatcher(this->language_extension_, sdfg_, node_.dataflow());
    dispatcher.dispatch(main_stream);
};

DataFlowDispatcher::DataFlowDispatcher(
    LanguageExtension& language_extension, const Function& sdfg, const data_flow::DataFlowGraph& data_flow_graph
)
    : language_extension_(language_extension), function_(sdfg), data_flow_graph_(data_flow_graph) {

      };

void DataFlowDispatcher::dispatch(PrettyPrinter& stream) {
    // Dispatch code nodes in topological order
    auto nodes = this->data_flow_graph_.topological_sort();
    for (auto& node : nodes) {
        if (auto tasklet = dynamic_cast<const data_flow::Tasklet*>(node)) {
            this->dispatch_tasklet(stream, *tasklet);
        } else if (auto libnode = dynamic_cast<const data_flow::LibraryNode*>(node)) {
            this->dispatch_library_node(stream, *libnode);
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

    stream << dst.data();
    stream << " = ";

    std::string rhs;
    if (symbolic::is_nullptr(symbolic::symbol(src.data())) || helpers::is_number(src.data())) {
        rhs += this->language_extension_.expression(symbolic::symbol(src.data()));
    } else {
        rhs += "&";
        rhs += src.data();
    }
    rhs += this->language_extension_.subset(function_, src_type, subset);

    // Check reinterpret_cast
    auto& res_type = types::infer_type(function_, src_type, subset);
    types::Pointer final_type(res_type);
    if (final_type == dst_type || symbolic::is_nullptr(symbolic::symbol(src.data()))) {
        stream << rhs;
    } else {
        stream << this->language_extension_.type_cast(rhs, dst_type);
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

    stream << dst.data();
    if (memlet.dst_conn() == "void") {
        stream << this->language_extension_.subset(function_, dst_type, memlet.subset());
    }
    stream << " = ";

    if (symbolic::is_nullptr(symbolic::symbol(src.data()))) {
        stream << this->language_extension_.expression(symbolic::__nullptr__());
    } else {
        stream << src.data();
    }
    if (memlet.src_conn() == "void") {
        stream << this->language_extension_.subset(function_, src_type, memlet.subset());
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

    std::unordered_map<std::string, const data_flow::Memlet*> in_edges;
    for (auto& iedge : this->data_flow_graph_.in_edges(tasklet)) {
        in_edges[iedge.dst_conn()] = &iedge;
    }
    for (auto& input : tasklet.inputs()) {
        if (in_edges.find(input.first) == in_edges.end()) {
            continue;
        }
        auto& iedge = *in_edges[input.first];
        auto& src = dynamic_cast<const data_flow::AccessNode&>(iedge.src());
        stream << this->language_extension_.declaration(input.first, input.second);
        const types::IType& type = this->function_.type(src.data());
        stream << " = " << src.data() << this->language_extension_.subset(function_, type, iedge.subset()) << ";";
        stream << std::endl;
    }
    stream << this->language_extension_.declaration(tasklet.output().first, tasklet.output().second);
    stream << ";" << std::endl;

    stream << std::endl;
    stream << tasklet.output().first << " = ";
    stream << this->language_extension_.tasklet(tasklet) << ";" << std::endl;
    stream << std::endl;

    // Write back
    for (auto& oedge : this->data_flow_graph_.out_edges(tasklet)) {
        auto& dst = dynamic_cast<const data_flow::AccessNode&>(oedge.dst());
        const types::IType& type = this->function_.type(dst.data());
        stream << dst.data() << this->language_extension_.subset(function_, type, oedge.subset()) << " = ";
        stream << oedge.src_conn();
        stream << ";" << std::endl;
    }

    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
};

void DataFlowDispatcher::dispatch_library_node(PrettyPrinter& stream, const data_flow::LibraryNode& libnode) {
    auto dispatcher_fn = LibraryNodeDispatcherRegistry::instance().get_library_node_dispatcher(libnode.code().value());
    if (dispatcher_fn) {
        auto dispatcher = dispatcher_fn(this->language_extension_, this->function_, this->data_flow_graph_, libnode);
        dispatcher->dispatch(stream);
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

} // namespace codegen
} // namespace sdfg
