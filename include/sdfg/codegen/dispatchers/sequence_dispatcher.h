#pragma once

#include "sdfg/codegen/dispatchers/node_dispatcher.h"

namespace sdfg {
namespace codegen {

class SequenceDispatcher : public NodeDispatcher {
   private:
    structured_control_flow::Sequence& node_;

   public:
    SequenceDispatcher(LanguageExtension& language_extension, Schedule& schedule,
                       structured_control_flow::Sequence& node, bool instrumented);

    void dispatch_node(PrettyPrinter& main_stream, PrettyPrinter& globals_stream,
                       PrettyPrinter& library_stream) override;
};
}  // namespace codegen
}  // namespace sdfg
