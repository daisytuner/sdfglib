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
        InstrumentationPlan& instrumentation_plan
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

    void dispatch_deref(PrettyPrinter& stream, const data_flow::Memlet& memlet);

    void dispatch_tasklet(PrettyPrinter& stream, const data_flow::Tasklet& tasklet);

    void dispatch_library_node(
        PrettyPrinter& stream,
        PrettyPrinter& globals_stream,
        CodeSnippetFactory& library_snippet_factory,
        const data_flow::LibraryNode& libnode
    );

public:
    DataFlowDispatcher(
        LanguageExtension& language_extension, const Function& function, const data_flow::DataFlowGraph& data_flow_graph
    );

    void dispatch(PrettyPrinter& stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory);
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

    virtual void
    dispatch(PrettyPrinter& stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory);

    virtual void dispatch_code(
        PrettyPrinter& stream,
        PrettyPrinter& globals_stream,
        CodeSnippetFactory& library_snippet_factory,
        std::vector<std::string>& inputs_by_order,
        std::vector<std::optional<std::string>>& outputs_by_order
    ) {}
};

} // namespace codegen
} // namespace sdfg
