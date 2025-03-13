#pragma once

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/codegen/dispatchers/node_dispatcher.h"
#include "sdfg/codegen/dispatchers/schedules/highway_dispatcher.h"
#include "sdfg/codegen/dispatchers/schedules/openmp_dispatcher.h"
#include "sdfg/codegen/dispatchers/sequence_dispatcher.h"

namespace sdfg {
namespace codegen {

class ForDispatcherSequential : public NodeDispatcher {
   private:
    structured_control_flow::For& node_;

   public:
    ForDispatcherSequential(LanguageExtension& language_extension, Schedule& schedule,
                            structured_control_flow::For& node, bool instrumented);

    void dispatch_node(PrettyPrinter& main_stream, PrettyPrinter& globals_stream,
                       PrettyPrinter& library_stream) override;

    void begin_instrumentation(PrettyPrinter& stream) override;

    void end_instrumentation(PrettyPrinter& stream) override;
};

class ForDispatcher : public NodeDispatcher {
   private:
    structured_control_flow::For& node_;

   public:
    ForDispatcher(LanguageExtension& language_extension, Schedule& schedule,
                  structured_control_flow::For& node, bool instrumented);

    void dispatch_node(PrettyPrinter& main_stream, PrettyPrinter& globals_stream,
                       PrettyPrinter& library_stream) override;

    void begin_instrumentation(PrettyPrinter& stream) override;

    void end_instrumentation(PrettyPrinter& stream) override;
};

}  // namespace codegen
}  // namespace sdfg
