#include "sdfg/codegen/dispatchers/node_dispatcher.h"

namespace sdfg {
namespace codegen {

NodeDispatcher::NodeDispatcher(LanguageExtension& language_extension, Schedule& schedule,
                               structured_control_flow::ControlFlowNode& node,
                               Instrumentation& instrumentation)
    : node_(node),
      language_extension_(language_extension),
      schedule_(schedule),
      instrumentation_(instrumentation) {};

bool NodeDispatcher::begin_node(PrettyPrinter& stream) { return false; };

void NodeDispatcher::end_node(PrettyPrinter& stream, bool applied) {

};

void NodeDispatcher::dispatch(PrettyPrinter& main_stream, PrettyPrinter& globals_stream,
                              PrettyPrinter& library_stream) {
    bool applied = begin_node(main_stream);

    if (this->instrumentation_.should_instrument(node_)) {
        this->instrumentation_.begin_instrumentation(node_, main_stream);
    }

    dispatch_node(main_stream, globals_stream, library_stream);

    if (this->instrumentation_.should_instrument(node_)) {
        this->instrumentation_.end_instrumentation(node_, main_stream);
    }

    end_node(main_stream, applied);
};

}  // namespace codegen
}  // namespace sdfg
