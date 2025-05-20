#pragma once

#include "sdfg/codegen/dispatchers/node_dispatcher.h"
#include "sdfg/structured_control_flow/map.h"

namespace sdfg {
namespace codegen {

class MapDispatcher : public NodeDispatcher {
   private:
    structured_control_flow::Map& node_;

   public:
    MapDispatcher(LanguageExtension& language_extension, Schedule& schedule,
                  structured_control_flow::Map& node, bool instrumented);

    void dispatch_node(PrettyPrinter& main_stream, PrettyPrinter& globals_stream,
                       PrettyPrinter& library_stream) override;
};
}  // namespace codegen
}  // namespace sdfg
