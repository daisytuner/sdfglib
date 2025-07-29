#pragma once

#include "sdfg/codegen/code_snippet_factory.h"
#include "sdfg/codegen/dispatchers/node_dispatcher.h"
#include "sdfg/structured_control_flow/map.h"

namespace sdfg {
namespace codegen {

class MapDispatcher : public NodeDispatcher {
private:
    structured_control_flow::Map& node_;

public:
    MapDispatcher(
        LanguageExtension& language_extension,
        StructuredSDFG& sdfg,
        structured_control_flow::Map& node,
        InstrumentationPlan& instrumentation_plan
    );

    void dispatch_node(
        PrettyPrinter& main_stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory
    ) override;
};

class SequentialMapDispatcher : public NodeDispatcher {
private:
    structured_control_flow::Map& node_;

public:
    SequentialMapDispatcher(
        LanguageExtension& language_extension,
        StructuredSDFG& sdfg,
        structured_control_flow::Map& node,
        InstrumentationPlan& instrumentation_plan
    );

    void dispatch_node(
        PrettyPrinter& main_stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory
    ) override;
};

class CPUParallelMapDispatcher : public NodeDispatcher {
private:
    structured_control_flow::Map& node_;

public:
    CPUParallelMapDispatcher(
        LanguageExtension& language_extension,
        StructuredSDFG& sdfg,
        structured_control_flow::Map& node,
        InstrumentationPlan& instrumentation_plan
    );

    void dispatch_node(
        PrettyPrinter& main_stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory
    ) override;
};

} // namespace codegen
} // namespace sdfg
