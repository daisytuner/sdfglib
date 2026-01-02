#pragma once

#include "sdfg/codegen/dispatchers/node_dispatcher_registry.h"
#include "sdfg/codegen/dispatchers/sequence_dispatcher.h"

namespace sdfg {
namespace codegen {

class WhileDispatcher : public NodeDispatcher {
private:
    structured_control_flow::While& node_;

public:
    WhileDispatcher(
        LanguageExtension& language_extension,
        StructuredSDFG& sdfg,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::While& node,
        InstrumentationPlan& instrumentation_plan,
        ArgCapturePlan& arg_capture_plan
    );

    void dispatch_node(
        PrettyPrinter& main_stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory
    ) override;

    InstrumentationInfo instrumentation_info() const override;
};

class BreakDispatcher : public NodeDispatcher {
private:
    structured_control_flow::Break& node_;

public:
    BreakDispatcher(
        LanguageExtension& language_extension,
        StructuredSDFG& sdfg,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::Break& node,
        InstrumentationPlan& instrumentation_plan,
        ArgCapturePlan& arg_capture_plan
    );

    void dispatch_node(
        PrettyPrinter& main_stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory
    ) override;
};

class ContinueDispatcher : public NodeDispatcher {
private:
    structured_control_flow::Continue& node_;

public:
    ContinueDispatcher(
        LanguageExtension& language_extension,
        StructuredSDFG& sdfg,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::Continue& node,
        InstrumentationPlan& instrumentation_plan,
        ArgCapturePlan& arg_capture_plan
    );

    void dispatch_node(
        PrettyPrinter& main_stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory
    ) override;
};

class ReturnDispatcher : public NodeDispatcher {
private:
    structured_control_flow::Return& node_;

public:
    ReturnDispatcher(
        LanguageExtension& language_extension,
        StructuredSDFG& sdfg,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::Return& node,
        InstrumentationPlan& instrumentation_plan,
        ArgCapturePlan& arg_capture_plan
    );

    void dispatch_node(
        PrettyPrinter& main_stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory
    ) override;
};

} // namespace codegen
} // namespace sdfg
