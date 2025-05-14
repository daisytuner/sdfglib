#include "sdfg/codegen/dispatchers/while_dispatcher.h"

namespace sdfg {
namespace codegen {

WhileDispatcher::WhileDispatcher(LanguageExtension& language_extension, Schedule& schedule,
                                 structured_control_flow::While& node, bool instrumented)
    : NodeDispatcher(language_extension, schedule, node, instrumented),
      node_(node){

      };

void WhileDispatcher::dispatch_node(PrettyPrinter& main_stream, PrettyPrinter& globals_stream,
                                    PrettyPrinter& library_stream) {
    main_stream << "while (1)" << std::endl;
    main_stream << "{" << std::endl;

    main_stream.setIndent(main_stream.indent() + 4);
    SequenceDispatcher dispatcher(language_extension_, schedule_, node_.root(), false);
    dispatcher.dispatch(main_stream, globals_stream, library_stream);
    main_stream.setIndent(main_stream.indent() - 4);

    main_stream << "}" << std::endl;
};

void WhileDispatcher::begin_instrumentation(PrettyPrinter& stream) {
    stream << "__daisy_instrument_enter();" << std::endl;
};

void WhileDispatcher::end_instrumentation(PrettyPrinter& stream) {
    stream << "__daisy_instrument_exit(";
    stream << "\"" << schedule_.sdfg().name() << "_" << node_.name() << "\", ";
    stream << "\"" << node_.debug_info().filename() << "\", ";
    stream << node_.debug_info().start_line() << ", ";
    stream << node_.debug_info().end_line() << ", ";
    stream << node_.debug_info().start_column() << ", ";
    stream << node_.debug_info().end_column() << ");" << std::endl;
};

BreakDispatcher::BreakDispatcher(LanguageExtension& language_extension, Schedule& schedule,
                                 structured_control_flow::Break& node, bool instrumented)
    : NodeDispatcher(language_extension, schedule, node, instrumented),
      node_(node){

      };

void BreakDispatcher::dispatch_node(PrettyPrinter& main_stream, PrettyPrinter& globals_stream,
                                    PrettyPrinter& library_stream) {
    main_stream << "break;" << std::endl;
};

ContinueDispatcher::ContinueDispatcher(LanguageExtension& language_extension, Schedule& schedule,
                                       structured_control_flow::Continue& node, bool instrumented)
    : NodeDispatcher(language_extension, schedule, node, instrumented),
      node_(node){

      };

void ContinueDispatcher::dispatch_node(PrettyPrinter& main_stream, PrettyPrinter& globals_stream,
                                       PrettyPrinter& library_stream) {
    main_stream << "continue;" << std::endl;
};

ReturnDispatcher::ReturnDispatcher(LanguageExtension& language_extension, Schedule& schedule,
                                   structured_control_flow::Return& node, bool instrumented)
    : NodeDispatcher(language_extension, schedule, node, instrumented),
      node_(node){

      };

void ReturnDispatcher::dispatch_node(PrettyPrinter& main_stream, PrettyPrinter& globals_stream,
                                     PrettyPrinter& library_stream) {
    main_stream << "return;" << std::endl;
};

}  // namespace codegen
}  // namespace sdfg
