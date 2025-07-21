#pragma once

#include "sdfg/codegen/dispatchers/node_dispatcher.h"
#include "sdfg/data_flow/library_node.h"

namespace sdfg {
namespace codegen {

class BlockDispatcher : public NodeDispatcher {
private:
    const structured_control_flow::Block& node_;

public:
    BlockDispatcher(
        LanguageExtension& language_extension,
        StructuredSDFG& sdfg,
        structured_control_flow::Block& node,
        Instrumentation& instrumentation
    );

    void dispatch_node(
        PrettyPrinter& main_stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory
    ) override;
};

class DataFlowDispatcher {
private:
    LanguageExtension& language_extension_;
    const Function& function_;
    const data_flow::DataFlowGraph& data_flow_graph_;

    void dispatch_ref(PrettyPrinter& stream, const data_flow::Memlet& memlet);

    void dispatch_tasklet(PrettyPrinter& stream, const data_flow::Tasklet& tasklet);

    void dispatch_library_node(PrettyPrinter& stream, const data_flow::LibraryNode& libnode);

public:
    DataFlowDispatcher(
        LanguageExtension& language_extension, const Function& function, const data_flow::DataFlowGraph& data_flow_graph
    );

    void dispatch(PrettyPrinter& stream);
};

class LibraryNodeDispatcher {
protected:
    LanguageExtension& language_extension_;
    const Function& function_;
    const data_flow::DataFlowGraph& data_flow_graph_;
    const data_flow::LibraryNode& node_;

public:
    LibraryNodeDispatcher(
        LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const data_flow::LibraryNode& node
    );

    virtual ~LibraryNodeDispatcher() = default;

    virtual void dispatch(PrettyPrinter& stream) = 0;
};

} // namespace codegen
} // namespace sdfg
