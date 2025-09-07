#include "sdfg/codegen/dispatchers/while_dispatcher.h"

namespace sdfg {
namespace codegen {

WhileDispatcher::WhileDispatcher(
    LanguageExtension& language_extension,
    StructuredSDFG& sdfg,
    structured_control_flow::While& node,
    InstrumentationPlan& instrumentation_plan
)
    : NodeDispatcher(language_extension, sdfg, node, instrumentation_plan), node_(node) {

      };

void WhileDispatcher::dispatch_node(
    PrettyPrinter& main_stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory
) {
    main_stream << "while (1)" << std::endl;
    main_stream << "{" << std::endl;

    main_stream.setIndent(main_stream.indent() + 4);
    SequenceDispatcher dispatcher(language_extension_, sdfg_, node_.root(), instrumentation_plan_);
    dispatcher.dispatch(main_stream, globals_stream, library_snippet_factory);
    main_stream.setIndent(main_stream.indent() - 4);

    main_stream << "}" << std::endl;
};

BreakDispatcher::BreakDispatcher(
    LanguageExtension& language_extension,
    StructuredSDFG& sdfg,
    structured_control_flow::Break& node,
    InstrumentationPlan& instrumentation_plan
)
    : NodeDispatcher(language_extension, sdfg, node, instrumentation_plan), node_(node) {

      };

void BreakDispatcher::dispatch_node(
    PrettyPrinter& main_stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory
) {
    main_stream << "break;" << std::endl;
};

ContinueDispatcher::ContinueDispatcher(
    LanguageExtension& language_extension,
    StructuredSDFG& sdfg,
    structured_control_flow::Continue& node,
    InstrumentationPlan& instrumentation_plan
)
    : NodeDispatcher(language_extension, sdfg, node, instrumentation_plan), node_(node) {

      };

void ContinueDispatcher::dispatch_node(
    PrettyPrinter& main_stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory
) {
    main_stream << "continue;" << std::endl;
};

ReturnDispatcher::ReturnDispatcher(
    LanguageExtension& language_extension,
    StructuredSDFG& sdfg,
    structured_control_flow::Return& node,
    InstrumentationPlan& instrumentation_plan
)
    : NodeDispatcher(language_extension, sdfg, node, instrumentation_plan), node_(node) {

      };

void ReturnDispatcher::dispatch_node(
    PrettyPrinter& main_stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory
) {
    if (node_.unreachable()) {
        main_stream << "/* unreachable return */" << std::endl;
        return;
    }
    main_stream << "return " << node_.data() << ";" << std::endl;
};

} // namespace codegen
} // namespace sdfg
