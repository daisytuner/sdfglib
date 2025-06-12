#include "sdfg/codegen/dispatchers/map_dispatcher.h"

#include "sdfg/codegen/dispatchers/sequence_dispatcher.h"

namespace sdfg {
namespace codegen {

MapDispatcher::MapDispatcher(LanguageExtension& language_extension, StructuredSDFG& sdfg,
                             structured_control_flow::Map& node, Instrumentation& instrumentation)
    : NodeDispatcher(language_extension, sdfg, node, instrumentation), node_(node) {

      };

void MapDispatcher::dispatch_node(PrettyPrinter& main_stream, PrettyPrinter& globals_stream,
                                  PrettyPrinter& library_stream) {
    auto dispatcher =
        MapDispatcherRegistry::instance().get_map_dispatcher(node_.schedule_type().value());
    if (dispatcher) {
        auto dispatcher_ptr = dispatcher(language_extension_, sdfg_, node_, instrumentation_);
        dispatcher_ptr->dispatch_node(main_stream, globals_stream, library_stream);
    } else {
        throw std::runtime_error("Unsupported map schedule type: " +
                                 std::string(node_.schedule_type().value()));
    }
};

SequentialMapDispatcher::SequentialMapDispatcher(LanguageExtension& language_extension,
                                                 StructuredSDFG& sdfg,
                                                 structured_control_flow::Map& node,
                                                 Instrumentation& instrumentation)
    : NodeDispatcher(language_extension, sdfg, node, instrumentation), node_(node) {

      };

void SequentialMapDispatcher::dispatch_node(PrettyPrinter& main_stream,
                                            PrettyPrinter& globals_stream,
                                            PrettyPrinter& library_stream) {
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
    SequenceDispatcher dispatcher(language_extension_, sdfg_, node_.root(), instrumentation_);
    dispatcher.dispatch(main_stream, globals_stream, library_stream);
    main_stream.setIndent(main_stream.indent() - 4);

    main_stream << "}" << std::endl;
};

CPUParallelMapDispatcher::CPUParallelMapDispatcher(LanguageExtension& language_extension,
                                                   StructuredSDFG& sdfg,
                                                   structured_control_flow::Map& node,
                                                   Instrumentation& instrumentation)
    : NodeDispatcher(language_extension, sdfg, node, instrumentation), node_(node) {

      };

void CPUParallelMapDispatcher::dispatch_node(PrettyPrinter& main_stream,
                                             PrettyPrinter& globals_stream,
                                             PrettyPrinter& library_stream) {
    main_stream << "#pragma omp parallel for" << std::endl;
    SequentialMapDispatcher dispatcher(language_extension_, sdfg_, node_, instrumentation_);
    dispatcher.dispatch(main_stream, globals_stream, library_stream);
};

void register_default_map_dispatchers() {
    MapDispatcherRegistry::instance().register_map_dispatcher(
        structured_control_flow::ScheduleType_Sequential.value(),
        [](LanguageExtension& language_extension, StructuredSDFG& sdfg,
           structured_control_flow::Map& node, Instrumentation& instrumentation) {
            return std::make_unique<SequentialMapDispatcher>(language_extension, sdfg, node,
                                                             instrumentation);
        });
    MapDispatcherRegistry::instance().register_map_dispatcher(
        structured_control_flow::ScheduleType_CPU_Parallel.value(),
        [](LanguageExtension& language_extension, StructuredSDFG& sdfg,
           structured_control_flow::Map& node, Instrumentation& instrumentation) {
            return std::make_unique<CPUParallelMapDispatcher>(language_extension, sdfg, node,
                                                              instrumentation);
        });
}

}  // namespace codegen
}  // namespace sdfg
