#pragma once

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/codegen/dispatchers/node_dispatcher_registry.h"
#include "sdfg/codegen/dispatchers/sequence_dispatcher.h"

namespace sdfg {
namespace codegen {

class ForDispatcher : public NodeDispatcher {
private:
    structured_control_flow::For& node_;

public:
    ForDispatcher(
        LanguageExtension& language_extension,
        StructuredSDFG& sdfg,
        structured_control_flow::For& node,
        InstrumentationPlan& instrumentation_plan
    );

    void dispatch_node(
        PrettyPrinter& main_stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory
    ) override;
};

} // namespace codegen
} // namespace sdfg
