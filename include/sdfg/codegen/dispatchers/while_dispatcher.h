#pragma once

#include "sdfg/codegen/dispatchers/node_dispatcher.h"
#include "sdfg/codegen/dispatchers/sequence_dispatcher.h"

namespace sdfg {
namespace codegen {

class WhileDispatcher : public NodeDispatcher {
   private:
    structured_control_flow::While& node_;

   public:
    WhileDispatcher(LanguageExtension& language_extension, Schedule& schedule,
                    structured_control_flow::While& node, Instrumentation& instrumentation);

    void dispatch_node(PrettyPrinter& main_stream, PrettyPrinter& globals_stream,
                       PrettyPrinter& library_stream) override;

};

class BreakDispatcher : public NodeDispatcher {
   private:
    structured_control_flow::Break& node_;

   public:
    BreakDispatcher(LanguageExtension& language_extension, Schedule& schedule,
                    structured_control_flow::Break& node, Instrumentation& instrumentation);

    void dispatch_node(PrettyPrinter& main_stream, PrettyPrinter& globals_stream,
                       PrettyPrinter& library_stream) override;
};

class ContinueDispatcher : public NodeDispatcher {
   private:
    structured_control_flow::Continue& node_;

   public:
    ContinueDispatcher(LanguageExtension& language_extension, Schedule& schedule,
                       structured_control_flow::Continue& node, Instrumentation& instrumentation);

    void dispatch_node(PrettyPrinter& main_stream, PrettyPrinter& globals_stream,
                       PrettyPrinter& library_stream) override;
};

class ReturnDispatcher : public NodeDispatcher {
   private:
    structured_control_flow::Return& node_;

   public:
    ReturnDispatcher(LanguageExtension& language_extension, Schedule& schedule,
                     structured_control_flow::Return& node, Instrumentation& instrumentation);

    void dispatch_node(PrettyPrinter& main_stream, PrettyPrinter& globals_stream,
                       PrettyPrinter& library_stream) override;
};

}  // namespace codegen
}  // namespace sdfg
