#pragma once

#include "sdfg/codegen/dispatchers/node_dispatcher.h"

namespace sdfg {
namespace codegen {

class ForDispatcher : public NodeDispatcher {
private:
    structured_control_flow::For& node_;

public:
    ForDispatcher(
        LanguageExtension& language_extension,
        StructuredSDFG& sdfg,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::For& node,
        InstrumentationPlan& instrumentation_plan,
        ArgCapturePlan& arg_capture_plan
    );

    void dispatch_node(
        PrettyPrinter& main_stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory
    ) override;

    InstrumentationInfo instrumentation_info() const override;
};

} // namespace codegen
} // namespace sdfg
