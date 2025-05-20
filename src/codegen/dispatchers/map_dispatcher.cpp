#include "sdfg/codegen/dispatchers/map_dispatcher.h"

namespace sdfg {
namespace codegen {

MapDispatcher::MapDispatcher(LanguageExtension& language_extension, Schedule& schedule,
                             structured_control_flow::Map& node, bool instrumented)
    : NodeDispatcher(language_extension, schedule, node, instrumented), node_(node) {

      };

void MapDispatcher::dispatch_node(PrettyPrinter& main_stream, PrettyPrinter& globals_stream,
                                  PrettyPrinter& library_stream) {
    // TODO: Implement MapDispatcher @Adrian
};

}  // namespace codegen
}  // namespace sdfg
