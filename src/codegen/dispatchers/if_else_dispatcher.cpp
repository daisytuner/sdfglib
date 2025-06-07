#include "sdfg/codegen/dispatchers/if_else_dispatcher.h"

#include "sdfg/codegen/dispatchers/sequence_dispatcher.h"

namespace sdfg {
namespace codegen {

IfElseDispatcher::IfElseDispatcher(LanguageExtension& language_extension, StructuredSDFG& sdfg,
                                   structured_control_flow::IfElse& node,
                                   Instrumentation& instrumentation)
    : NodeDispatcher(language_extension, sdfg, node, instrumentation), node_(node) {

      };

void IfElseDispatcher::dispatch_node(PrettyPrinter& main_stream, PrettyPrinter& globals_stream,
                                     PrettyPrinter& library_stream) {
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
        SequenceDispatcher dispatcher(language_extension_, sdfg_, child.first, instrumentation_);
        dispatcher.dispatch(main_stream, globals_stream, library_stream);
        main_stream.setIndent(main_stream.indent() - 4);

        main_stream << "}" << std::endl;
    }
};

}  // namespace codegen
}  // namespace sdfg
