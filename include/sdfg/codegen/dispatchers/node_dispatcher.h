#pragma once

#include "sdfg/analysis/analysis.h"
#include "sdfg/codegen/code_snippet_factory.h"
#include "sdfg/codegen/instrumentation/arg_capture_plan.h"
#include "sdfg/codegen/instrumentation/instrumentation_plan.h"
#include "sdfg/codegen/language_extension.h"
#include "sdfg/codegen/utils.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_sdfg.h"

namespace sdfg {
namespace codegen {

class NodeDispatcher {
private:
    structured_control_flow::ControlFlowNode& node_;

protected:
    LanguageExtension& language_extension_;

    StructuredSDFG& sdfg_;

    InstrumentationPlan& instrumentation_plan_;

    ArgCapturePlan& arg_capture_plan_;

    analysis::AnalysisManager& analysis_manager_;

    virtual bool begin_node(PrettyPrinter& stream);

    virtual void end_node(PrettyPrinter& stream, bool has_declaration);

    /**
     * Bad design. We already have fields in this class and have to bind an instance to a single node.
     * Just set a field to sth. of the instrumentation plan where the node can register more details. But this is a
     * breaking change
     */
    virtual InstrumentationInfo instrumentation_info() const;

public:
    NodeDispatcher(
        LanguageExtension& language_extension,
        StructuredSDFG& sdfg,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::ControlFlowNode& node,
        InstrumentationPlan& instrumentation_plan,
        ArgCapturePlan& arg_capture_plan
    );

    virtual ~NodeDispatcher() = default;

    virtual void dispatch_node(
        PrettyPrinter& main_stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory
    ) = 0;

    virtual void
    dispatch(PrettyPrinter& main_stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory);
};

} // namespace codegen
} // namespace sdfg
