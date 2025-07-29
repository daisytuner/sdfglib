#include "sdfg/codegen/dispatchers/node_dispatcher.h"

namespace sdfg {
namespace codegen {

NodeDispatcher::NodeDispatcher(
    LanguageExtension& language_extension,
    StructuredSDFG& sdfg,
    structured_control_flow::ControlFlowNode& node,
    InstrumentationPlan& instrumentation_plan
)
    : node_(node), language_extension_(language_extension), sdfg_(sdfg), instrumentation_plan_(instrumentation_plan) {};

bool NodeDispatcher::begin_node(PrettyPrinter& stream) { return false; };

void NodeDispatcher::end_node(PrettyPrinter& stream, bool applied) {

};

void NodeDispatcher::
    dispatch(PrettyPrinter& main_stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory) {
    bool applied = begin_node(main_stream);

    if (this->instrumentation_plan_.should_instrument(node_)) {
        this->instrumentation_plan_.begin_instrumentation(node_, main_stream);
    }

    dispatch_node(main_stream, globals_stream, library_snippet_factory);

    if (this->instrumentation_plan_.should_instrument(node_)) {
        this->instrumentation_plan_.end_instrumentation(node_, main_stream);
    }

    end_node(main_stream, applied);
};

} // namespace codegen
} // namespace sdfg
