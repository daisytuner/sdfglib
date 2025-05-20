#include "sdfg/codegen/dispatchers/map_dispatcher.h"

#include "sdfg/codegen/dispatchers/sequence_dispatcher.h"

namespace sdfg {
namespace codegen {

MapDispatcher::MapDispatcher(LanguageExtension& language_extension, Schedule& schedule,
                             structured_control_flow::Map& node, bool instrumented)
    : NodeDispatcher(language_extension, schedule, node, instrumented), node_(node) {

      };

void MapDispatcher::dispatch_node(PrettyPrinter& main_stream, PrettyPrinter& globals_stream,
                                  PrettyPrinter& library_stream) {
    LoopSchedule loop_schedule = this->schedule_.loop_schedule(&this->node_);
    if (loop_schedule == LoopSchedule::VECTORIZATION) {
        main_stream << "#pragma omp simd" << std::endl;
    }
    if (loop_schedule == LoopSchedule::MULTICORE) {
        main_stream << "#pragma omp parallel for" << std::endl;
    }

    main_stream << "for";
    main_stream << "(";
    main_stream << node_.indvar()->get_name();
    main_stream << " = 0; ";
    main_stream << node_.indvar()->get_name();
    main_stream << " < ";
    main_stream << language_extension_.expression(node_.num_iterations());
    main_stream << "; ";
    main_stream << node_.indvar()->get_name();
    main_stream << "++)" << std::endl;
    main_stream << "{" << std::endl;

    main_stream.setIndent(main_stream.indent() + 4);
    SequenceDispatcher dispatcher(language_extension_, schedule_, node_.root(), instrumented_);
    dispatcher.dispatch(main_stream, globals_stream, library_stream);
    main_stream.setIndent(main_stream.indent() - 4);

    main_stream << "}" << std::endl;
};

}  // namespace codegen
}  // namespace sdfg
