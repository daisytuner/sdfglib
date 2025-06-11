#include "sdfg/codegen/dispatchers/block_dispatcher.h"

#include "sdfg/codegen/dispatchers/library_nodes/thread_barrier_dispatcher.h"

namespace sdfg {
namespace codegen {

BlockDispatcher::BlockDispatcher(LanguageExtension& language_extension, StructuredSDFG& sdfg,
                                 structured_control_flow::Block& node,
                                 Instrumentation& instrumentation)
    : sdfg::codegen::NodeDispatcher(language_extension, sdfg, node, instrumentation), node_(node) {

      };

void BlockDispatcher::dispatch_node(PrettyPrinter& main_stream, PrettyPrinter& globals_stream,
                                    PrettyPrinter& library_stream) {
    if (node_.dataflow().nodes().empty()) {
        return;
    }

    DataFlowDispatcher dispatcher(this->language_extension_, sdfg_, node_.dataflow());
    dispatcher.dispatch(main_stream);
};

DataFlowDispatcher::DataFlowDispatcher(LanguageExtension& language_extension, const Function& sdfg,
                                       const data_flow::DataFlowGraph& data_flow_graph)
    : language_extension_(language_extension), function_(sdfg), data_flow_graph_(data_flow_graph) {

      };

void DataFlowDispatcher::dispatch(PrettyPrinter& stream) {
    // Dispatch code nodes in topological order
    auto nodes = this->data_flow_graph_.topological_sort();
    for (auto& node : nodes) {
        if (auto tasklet = dynamic_cast<const data_flow::Tasklet*>(node)) {
            this->dispatch_tasklet(stream, *tasklet);
        } else if (auto access_node = dynamic_cast<const data_flow::AccessNode*>(node)) {
            for (auto& edge : this->data_flow_graph_.out_edges(*access_node)) {
                if (edge.dst_conn() == "refs" || edge.src_conn() == "refs") {
                    this->dispatch_ref(stream, edge);
                }
            }
        } else if (auto libnode = dynamic_cast<const data_flow::LibraryNode*>(node)) {
            this->dispatch_library_node(stream, *libnode);
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

    const types::IType* final_src_type = &src_type;
    const types::IType* final_dst_type = &dst_type;

    stream << dst.data();
    if (memlet.dst_conn() == "void") {
        stream << this->language_extension_.subset(function_, dst_type, memlet.subset());

        final_dst_type = &types::infer_type(function_, *final_dst_type, memlet.subset());
    }
    stream << " = ";

    std::string rhs;
    bool allocated_src_type = false;
    if (memlet.src_conn() == "void") {
        // Infer the dereferenced type
        auto subset = memlet.subset();
        const types::IType* dereferenced_type =
            &types::infer_type(function_, *final_src_type, subset);

        // Function are incomplete types, so we need to remove one level of indirection
        if (dynamic_cast<const types::Function*>(dereferenced_type)) {
            subset.pop_back();
        } else {
            rhs += "&";
        }
        rhs += src.data();
        rhs += this->language_extension_.subset(function_, *final_src_type, subset);

        allocated_src_type = true;
        final_src_type = new types::Pointer(*dereferenced_type);
    } else {
        if (symbolic::is_pointer(symbolic::symbol(src.data()))) {
            rhs += this->language_extension_.expression(symbolic::symbol(src.data()));
        } else {
            rhs += src.data();
        }
    }

    if (*final_dst_type == *final_src_type || symbolic::is_pointer(symbolic::symbol(src.data()))) {
        stream << rhs;
    } else {
        stream << this->language_extension_.type_cast(rhs, *final_dst_type);
    }

    if (allocated_src_type) {
        delete final_src_type;
    }

    stream << ";";
    stream << std::endl;

    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
};

void DataFlowDispatcher::dispatch_tasklet(PrettyPrinter& stream,
                                          const data_flow::Tasklet& tasklet) {
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
        stream << " = " << src.data()
               << this->language_extension_.subset(function_, type, iedge.subset()) << ";";
        stream << std::endl;
    }
    stream << this->language_extension_.declaration(tasklet.output().first,
                                                    tasklet.output().second);
    stream << ";" << std::endl;

    stream << std::endl;
    stream << tasklet.output().first << " = ";
    stream << this->language_extension_.tasklet(tasklet) << ";" << std::endl;
    stream << std::endl;

    // Write back
    for (auto& oedge : this->data_flow_graph_.out_edges(tasklet)) {
        auto& dst = dynamic_cast<const data_flow::AccessNode&>(oedge.dst());
        const types::IType& type = this->function_.type(dst.data());
        stream << dst.data() << this->language_extension_.subset(function_, type, oedge.subset())
               << " = ";
        stream << oedge.src_conn();
        stream << ";" << std::endl;
    }

    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
};

void DataFlowDispatcher::dispatch_library_node(PrettyPrinter& stream,
                                               const data_flow::LibraryNode& libnode) {
    auto dispatcher_fn = LibraryNodeDispatcherRegistry::instance().get_library_node_dispatcher(
        libnode.code().value());
    if (dispatcher_fn) {
        auto dispatcher = dispatcher_fn(this->language_extension_, this->function_,
                                        this->data_flow_graph_, libnode);
        dispatcher->dispatch(stream);
    } else {
        throw std::runtime_error("No library node dispatcher found for library node code: " +
                                 std::string(libnode.code().value()));
    }
};

void LibraryNodeDispatcherRegistry::register_default_library_node_dispatchers() {
    LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        data_flow::LibraryNodeCode{"barrier_local"}.value(),
        [](LanguageExtension& language_extension, const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph, const data_flow::LibraryNode& node) {
            return std::make_unique<ThreadBarrierDispatcher>(
                language_extension, function, data_flow_graph,
                dynamic_cast<const data_flow::ThreadBarrierNode&>(node));
        });
};

LibraryNodeDispatcher::LibraryNodeDispatcher(LanguageExtension& language_extension,
                                             const Function& function,
                                             const data_flow::DataFlowGraph& data_flow_graph,
                                             const data_flow::LibraryNode& node)
    : language_extension_(language_extension),
      function_(function),
      data_flow_graph_(data_flow_graph),
      node_(node) {};

}  // namespace codegen
}  // namespace sdfg
