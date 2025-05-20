#pragma once

#include "sdfg/codegen/dispatchers/node_dispatcher.h"

namespace sdfg {
namespace codegen {

class IfElseDispatcher : public NodeDispatcher {
   private:
    structured_control_flow::IfElse& node_;

   public:
    IfElseDispatcher(LanguageExtension& language_extension, Schedule& schedule,
                     structured_control_flow::IfElse& node, Instrumentation& instrumentation);

    void dispatch_node(PrettyPrinter& main_stream, PrettyPrinter& globals_stream,
                       PrettyPrinter& library_stream) override;
};
}  // namespace codegen
}  // namespace sdfg
