#pragma once

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/codegen/dispatchers/node_dispatcher_registry.h"
#include "sdfg/codegen/dispatchers/sequence_dispatcher.h"

namespace sdfg {
namespace codegen {

class ForDispatcher : public NodeDispatcher {
   private:
    structured_control_flow::For& node_;

   public:
    ForDispatcher(LanguageExtension& language_extension, StructuredSDFG& sdfg,
                  structured_control_flow::For& node, Instrumentation& instrumentation);

    void dispatch_node(PrettyPrinter& main_stream, PrettyPrinter& globals_stream,
                       PrettyPrinter& library_stream) override;
};

}  // namespace codegen
}  // namespace sdfg
