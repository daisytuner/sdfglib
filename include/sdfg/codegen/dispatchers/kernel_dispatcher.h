#pragma once

#include "sdfg/codegen/dispatchers/node_dispatcher.h"
#include "sdfg/codegen/dispatchers/sequence_dispatcher.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace codegen {

class KernelDispatcher : public NodeDispatcher {
   private:
    structured_control_flow::Kernel& node_;

   public:
    KernelDispatcher(LanguageExtension& language_extension, Schedule& schedule,
                     structured_control_flow::Kernel& node, bool instrumented);

    void dispatch_node(PrettyPrinter& main_stream, PrettyPrinter& globals_stream,
                       PrettyPrinter& library_stream) override;
};
}  // namespace codegen
}  // namespace sdfg
