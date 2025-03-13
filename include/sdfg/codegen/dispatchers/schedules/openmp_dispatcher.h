#pragma once

#include "sdfg/codegen/dispatchers/node_dispatcher.h"
#include "sdfg/codegen/dispatchers/sequence_dispatcher.h"

namespace sdfg {
namespace codegen {

class OpenMPDispatcher : public NodeDispatcher {
   private:
    structured_control_flow::For& node_;

   public:
    OpenMPDispatcher(LanguageExtension& language_extension, Schedule& schedule,
                     structured_control_flow::For& node, bool instrumented);

    void dispatch_node(PrettyPrinter& main_stream, PrettyPrinter& globals_stream,
                       PrettyPrinter& library_stream) override;
};

}  // namespace codegen
}  // namespace sdfg
