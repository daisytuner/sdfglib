#pragma once

#include "sdfg/codegen/dispatchers/node_dispatcher.h"

namespace sdfg {
namespace codegen {

class SequenceDispatcher : public NodeDispatcher {
private:
    structured_control_flow::Sequence& node_;

public:
    SequenceDispatcher(
        LanguageExtension& language_extension,
        StructuredSDFG& sdfg,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::Sequence& node,
        InstrumentationPlan& instrumentation_plan,
        ArgCapturePlan& arg_capture_plan
    );

    void dispatch_node(
        PrettyPrinter& main_stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory
    ) override;
};
} // namespace codegen
} // namespace sdfg
