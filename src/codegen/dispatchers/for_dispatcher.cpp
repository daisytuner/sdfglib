#include "sdfg/codegen/dispatchers/for_dispatcher.h"

namespace sdfg {
namespace codegen {

ForDispatcherSequential::ForDispatcherSequential(LanguageExtension& language_extension,
                                                 Schedule& schedule,
                                                 structured_control_flow::For& node,
                                                 bool instrumented)
    : NodeDispatcher(language_extension, schedule, node, instrumented),
      node_(node){

      };

void ForDispatcherSequential::dispatch_node(PrettyPrinter& main_stream,
                                            PrettyPrinter& globals_stream,
                                            PrettyPrinter& library_stream) {
    main_stream << "for";
    main_stream << "(";
    main_stream << node_.indvar()->get_name();
    main_stream << " = ";
    main_stream << language_extension_.expression(node_.init());
    main_stream << ";";
    main_stream << language_extension_.expression(node_.condition());
    main_stream << ";";
    main_stream << node_.indvar()->get_name();
    main_stream << " = ";
    main_stream << language_extension_.expression(node_.update());
    main_stream << ")" << std::endl;
    main_stream << "{" << std::endl;

    main_stream.setIndent(main_stream.indent() + 4);
    SequenceDispatcher dispatcher(language_extension_, schedule_, node_.root(), instrumented_);
    dispatcher.dispatch(main_stream, globals_stream, library_stream);
    main_stream.setIndent(main_stream.indent() - 4);

    main_stream << "}" << std::endl;
};

void ForDispatcherSequential::begin_instrumentation(PrettyPrinter& stream){};

void ForDispatcherSequential::end_instrumentation(PrettyPrinter& stream){};

ForDispatcher::ForDispatcher(LanguageExtension& language_extension, Schedule& schedule,
                             structured_control_flow::For& node, bool instrumented)
    : NodeDispatcher(language_extension, schedule, node, instrumented),
      node_(node){

      };

void ForDispatcher::dispatch_node(PrettyPrinter& main_stream, PrettyPrinter& globals_stream,
                                  PrettyPrinter& library_stream) {
    auto loop_schedule = schedule_.loop_schedule(&node_);
    switch (loop_schedule) {
        case LoopSchedule::SEQUENTIAL: {
            ForDispatcherSequential dispatcher(language_extension_, schedule_, node_, false);
            dispatcher.dispatch(main_stream, globals_stream, library_stream);
            break;
        }
        case LoopSchedule::VECTORIZATION: {
            HighwayDispatcher dispatcher(language_extension_, schedule_, node_, false);
            dispatcher.dispatch(main_stream, globals_stream, library_stream);
            break;
        }
        case LoopSchedule::MULTICORE: {
            OpenMPDispatcher dispatcher(language_extension_, schedule_, node_, false);
            dispatcher.dispatch(main_stream, globals_stream, library_stream);
            break;
        }
    }
};

void ForDispatcher::begin_instrumentation(PrettyPrinter& stream) {
    stream << "__daisy_instrument_enter();" << std::endl;
};

void ForDispatcher::end_instrumentation(PrettyPrinter& stream) {
    stream << "__daisy_instrument_exit(";
    stream << "\"" << schedule_.sdfg().name() << "_" << node_.name() << "\", ";
    stream << "\"" << node_.debug_info().filename() << "\", ";
    stream << node_.debug_info().start_line() << ", ";
    stream << node_.debug_info().end_line() << ", ";
    stream << node_.debug_info().start_column() << ", ";
    stream << node_.debug_info().end_column() << ");" << std::endl;
};

}  // namespace codegen
}  // namespace sdfg
