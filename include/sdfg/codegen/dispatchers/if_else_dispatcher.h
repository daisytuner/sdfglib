#pragma once

#include "sdfg/codegen/dispatchers/node_dispatcher_registry.h"

namespace sdfg {
namespace codegen {

class IfElseDispatcher : public NodeDispatcher {
private:
    structured_control_flow::IfElse& node_;

public:
    IfElseDispatcher(
        LanguageExtension& language_extension,
        StructuredSDFG& sdfg,
        structured_control_flow::IfElse& node,
        Instrumentation& instrumentation
    );

    void dispatch_node(
        PrettyPrinter& main_stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory
    ) override;
};
} // namespace codegen
} // namespace sdfg
