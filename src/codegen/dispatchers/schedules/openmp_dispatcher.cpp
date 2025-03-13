#include "sdfg/codegen/dispatchers/schedules/openmp_dispatcher.h"

namespace sdfg {
namespace codegen {

OpenMPDispatcher::OpenMPDispatcher(LanguageExtension& language_extension, Schedule& schedule,
                                   structured_control_flow::For& node, bool instrumented)
    : NodeDispatcher(language_extension, schedule, node, instrumented),
      node_(node){

      };

void OpenMPDispatcher::dispatch_node(PrettyPrinter& main_stream, PrettyPrinter& globals_stream,
                                     PrettyPrinter& library_stream) {
    main_stream << "#pragma omp parallel for private(" << node_.indvar()->get_name() << ")"
                << std::endl;

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
    SequenceDispatcher dispatcher(language_extension_, schedule_, node_.root(), false);
    dispatcher.dispatch(main_stream, globals_stream, library_stream);
    main_stream.setIndent(main_stream.indent() - 4);

    main_stream << "}" << std::endl;
};

}  // namespace codegen
}  // namespace sdfg
