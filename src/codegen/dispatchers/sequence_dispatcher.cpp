#include "sdfg/codegen/dispatchers/sequence_dispatcher.h"

#include "sdfg/codegen/dispatchers/node_dispatcher_registry.h"

namespace sdfg {
namespace codegen {

SequenceDispatcher::SequenceDispatcher(
    LanguageExtension& language_extension,
    StructuredSDFG& sdfg,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Sequence& node,
    InstrumentationPlan& instrumentation_plan
)
    : NodeDispatcher(language_extension, sdfg, analysis_manager, node, instrumentation_plan), node_(node) {

      };

void SequenceDispatcher::dispatch_node(
    PrettyPrinter& main_stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory
) {
    for (size_t i = 0; i < node_.size(); i++) {
        auto child = node_.at(i);

        // Node
        main_stream.setIndent(main_stream.indent() + 4);
        auto dispatcher =
            create_dispatcher(language_extension_, sdfg_, analysis_manager_, child.first, instrumentation_plan_);
        dispatcher->dispatch(main_stream, globals_stream, library_snippet_factory);
        main_stream.setIndent(main_stream.indent() - 4);

        // Transition
        if (!child.second.assignments().empty()) {
            main_stream << "{" << std::endl;
            main_stream.setIndent(main_stream.indent() + 4);
            for (auto assign : child.second.assignments()) {
                main_stream << language_extension_.expression(assign.first) << " = "
                            << language_extension_.expression(assign.second) << ";" << std::endl;
            }
            main_stream.setIndent(main_stream.indent() - 4);
            main_stream << "}" << std::endl;
        }
    }
};

} // namespace codegen
} // namespace sdfg
