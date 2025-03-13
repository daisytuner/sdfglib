#include "sdfg/codegen/dispatchers/block_dispatcher.h"

namespace sdfg {
namespace codegen {

void DataFlowDispatcher::dispatch_tasklet(PrettyPrinter& stream,
                                          const data_flow::Tasklet& tasklet) {
    if (tasklet.is_conditional()) {
        stream << "if (" << language_extension_.expression(tasklet.condition()) << ") ";
    }
    stream << "{" << std::endl;
    stream.setIndent(stream.indent() + 4);

    // Connector declarations
    for (auto& iedge : this->data_flow_graph_.in_edges(tasklet)) {
        auto& src = dynamic_cast<const data_flow::AccessNode&>(iedge.src());

        const types::Scalar& conn_type = tasklet.input_type(iedge.dst_conn());
        stream << this->language_extension_.declaration(iedge.dst_conn(), conn_type);

        if (symbolic::is_nvptx(symbolic::symbol(src.data()))) {
            stream << " = " << src.data() << ";";
        } else {
            const types::IType& type = this->function_.type(src.data());
            stream << " = " << src.data()
                   << this->language_extension_.subset(function_, type, iedge.subset()) << ";";
        }
        stream << std::endl;
    }
    for (auto& oedge : this->data_flow_graph_.out_edges(tasklet)) {
        const types::Scalar& conn_type = tasklet.output_type(oedge.src_conn());
        stream << this->language_extension_.declaration(oedge.src_conn(), conn_type) << ";"
               << std::endl;
    }

    stream << std::endl;
    stream << tasklet.output(0).first << " = ";
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

void DataFlowDispatcher::dispatch_ref(PrettyPrinter& stream, const data_flow::Memlet& memlet) {
    stream << "{" << std::endl;
    stream.setIndent(stream.indent() + 4);

    auto dst_access_node = static_cast<const data_flow::AccessNode*>(&memlet.dst());
    auto dst_type =
        static_cast<const types::Pointer*>(&this->function_.type(dst_access_node->data()));

    stream << dst_access_node->data();

    // Case 1: Dereference dst
    if (memlet.dst_conn() == "void") {
        stream << this->language_extension_.subset(function_, *dst_type, memlet.subset());

        dst_type = static_cast<const types::Pointer*>(
            &types::infer_type(function_, *dst_type, memlet.subset()));
    }

    stream << " = ";

    // src is raw pointer
    auto src_access_node = static_cast<const data_flow::AccessNode*>(&memlet.src());
    if (symbolic::is_pointer(symbolic::symbol(src_access_node->data()))) {
        stream << language_extension_.expression(symbolic::symbol(src_access_node->data()));
    } else {
        auto src_type = &this->function_.type(src_access_node->data());

        // Case 2: Dereference src
        std::string rhs;
        if (memlet.src_conn() == "void") {
            rhs += "&";
            rhs += src_access_node->data();
            rhs += this->language_extension_.subset(function_, *src_type, memlet.subset());

            src_type = new types::Pointer(types::infer_type(function_, *src_type, memlet.subset()));
        } else {
            rhs = src_access_node->data();
        }

        if (*dst_type == *src_type) {
            stream << rhs;
        } else {
            stream << this->language_extension_.type_cast(rhs, *dst_type);
        }

        if (memlet.src_conn() == "void") {
            delete src_type;
        }
    }

    stream << ";";
    stream << std::endl;

    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
};

void DataFlowDispatcher::dispatch_library_node(PrettyPrinter& stream,
                                               const data_flow::LibraryNode& libnode) {
    stream << "{" << std::endl;
    stream.setIndent(stream.indent() + 4);

    // Connector declarations
    for (auto& iedge : this->data_flow_graph_.in_edges(libnode)) {
        auto& src = dynamic_cast<const data_flow::AccessNode&>(iedge.src());

        const types::Scalar& conn_type = libnode.input_type(iedge.dst_conn());
        stream << this->language_extension_.declaration(iedge.dst_conn(), conn_type);

        if (symbolic::is_nvptx(symbolic::symbol(src.data()))) {
            stream << " = " << src.data() << ";";
        } else {
            const types::IType& type = this->function_.type(src.data());
            stream << " = " << src.data()
                   << this->language_extension_.subset(function_, type, iedge.subset()) << ";";
        }
        stream << std::endl;
    }
    for (auto& oedge : this->data_flow_graph_.out_edges(libnode)) {
        const types::Scalar& conn_type = libnode.output_type(oedge.src_conn());
        stream << this->language_extension_.declaration(oedge.src_conn(), conn_type) << ";"
               << std::endl;
    }

    stream << std::endl;
    stream << this->language_extension_.library_node(libnode);
    stream << std::endl;

    // Write back
    for (auto& oedge : this->data_flow_graph_.out_edges(libnode)) {
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

DataFlowDispatcher::DataFlowDispatcher(LanguageExtension& language_extension, const Function& sdfg,
                                       const data_flow::DataFlowGraph& data_flow_graph)
    : language_extension_(language_extension),
      function_(sdfg),
      data_flow_graph_(data_flow_graph){

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

BlockDispatcher::BlockDispatcher(LanguageExtension& language_extension, Schedule& schedule,
                                 structured_control_flow::Block& node, bool instrumented)
    : NodeDispatcher(language_extension, schedule, node, instrumented),
      node_(node){

      };

void BlockDispatcher::dispatch_node(PrettyPrinter& main_stream, PrettyPrinter& globals_stream,
                                    PrettyPrinter& library_stream) {
    if (node_.dataflow().nodes().empty()) {
        return;
    }

    auto& sdfg = schedule_.sdfg();
    DataFlowDispatcher dispatcher(this->language_extension_, sdfg, node_.dataflow());
    dispatcher.dispatch(main_stream);
};

}  // namespace codegen
}  // namespace sdfg
