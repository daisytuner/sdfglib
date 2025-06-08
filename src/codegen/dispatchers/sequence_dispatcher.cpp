#include "sdfg/codegen/dispatchers/sequence_dispatcher.h"

#include "sdfg/codegen/dispatchers/node_dispatcher_registry.h"

namespace sdfg {
namespace codegen {

SequenceDispatcher::SequenceDispatcher(LanguageExtension& language_extension, StructuredSDFG& sdfg,
                                       structured_control_flow::Sequence& node,
                                       Instrumentation& instrumentation)
    : NodeDispatcher(language_extension, sdfg, node, instrumentation), node_(node) {

      };

void SequenceDispatcher::dispatch_node(PrettyPrinter& main_stream, PrettyPrinter& globals_stream,
                                       PrettyPrinter& library_stream) {
    for (size_t i = 0; i < node_.size(); i++) {
        auto child = node_.at(i);

        // Node
        main_stream.setIndent(main_stream.indent() + 4);
        auto dispatcher =
            create_dispatcher(language_extension_, sdfg_, child.first, instrumentation_);
        dispatcher->dispatch(main_stream, globals_stream, library_stream);
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

}  // namespace codegen
}  // namespace sdfg
