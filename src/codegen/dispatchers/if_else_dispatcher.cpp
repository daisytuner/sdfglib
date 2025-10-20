#include "sdfg/codegen/dispatchers/if_else_dispatcher.h"

#include "sdfg/codegen/dispatchers/sequence_dispatcher.h"

namespace sdfg {
namespace codegen {

IfElseDispatcher::IfElseDispatcher(
    LanguageExtension& language_extension,
    StructuredSDFG& sdfg,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::IfElse& node,
    InstrumentationPlan& instrumentation_plan
)
    : NodeDispatcher(language_extension, sdfg, analysis_manager, node, instrumentation_plan), node_(node) {

      };

void IfElseDispatcher::dispatch_node(
    PrettyPrinter& main_stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory
) {
    for (size_t i = 0; i < node_.size(); i++) {
        auto child = node_.at(i);

        if (i == 0) {
            main_stream << "if";
        } else {
            main_stream << "else if";
        }
        main_stream << "(";
        main_stream << language_extension_.expression(child.second);
        main_stream << ")";
        main_stream << std::endl;

        main_stream << "{" << std::endl;

        main_stream.setIndent(main_stream.indent() + 4);
        SequenceDispatcher dispatcher(language_extension_, sdfg_, analysis_manager_, child.first, instrumentation_plan_);
        dispatcher.dispatch(main_stream, globals_stream, library_snippet_factory);
        main_stream.setIndent(main_stream.indent() - 4);

        main_stream << "}" << std::endl;
    }
};

} // namespace codegen
} // namespace sdfg
