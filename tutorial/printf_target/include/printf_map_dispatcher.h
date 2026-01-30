#pragma once

#include <sdfg/analysis/analysis.h>
#include <sdfg/codegen/dispatchers/node_dispatcher.h>
#include <sdfg/codegen/instrumentation/instrumentation_info.h>
#include <sdfg/structured_control_flow/map.h>

namespace sdfg {
namespace printf_target {

/**
 * @brief Map dispatcher that generates printf statements instead of kernel code
 *
 * This dispatcher replaces actual kernel generation with printf statements
 * that trace the execution flow, including:
 * - Map entry/exit with iteration bounds
 * - Iteration variable values
 * - Arguments being accessed
 */
class PrintfMapDispatcher : public codegen::NodeDispatcher {
private:
    structured_control_flow::Map& node_;

    /// Generates printf statements for the map body
    void dispatch_printf_body(
        codegen::PrettyPrinter& stream,
        codegen::CodeSnippetFactory& library_snippet_factory,
        codegen::PrettyPrinter& globals_stream
    );

public:
    PrintfMapDispatcher(
        codegen::LanguageExtension& language_extension,
        StructuredSDFG& sdfg,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::Map& node,
        codegen::InstrumentationPlan& instrumentation_plan,
        codegen::ArgCapturePlan& arg_capture_plan
    );

    /// Generates the printf-based tracing code for this map
    void dispatch_node(
        codegen::PrettyPrinter& main_stream,
        codegen::PrettyPrinter& globals_stream,
        codegen::CodeSnippetFactory& library_snippet_factory
    ) override;

    /// Returns instrumentation info for this map
    virtual codegen::InstrumentationInfo instrumentation_info() const override;
};

} // namespace printf_target
} // namespace sdfg
