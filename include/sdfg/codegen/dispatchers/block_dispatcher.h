#pragma once

#include "sdfg/codegen/dispatchers/node_dispatcher.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/utils.h"

namespace sdfg {
namespace codegen {

class DataFlowDispatcher {
   private:
    LanguageExtension& language_extension_;
    const Function& function_;
    const data_flow::DataFlowGraph& data_flow_graph_;

    void dispatch_tasklet(PrettyPrinter& stream, const data_flow::Tasklet& tasklet);

    void dispatch_ref(PrettyPrinter& stream, const data_flow::Memlet& memlet);

    void dispatch_library_node(PrettyPrinter& stream, const data_flow::LibraryNode& libnode);

   public:
    DataFlowDispatcher(LanguageExtension& language_extension, const Function& sdfg,
                       const data_flow::DataFlowGraph& data_flow_graph);

    void dispatch(PrettyPrinter& stream);
};

class BlockDispatcher : public NodeDispatcher {
   private:
    const structured_control_flow::Block& node_;

   public:
    BlockDispatcher(LanguageExtension& language_extension, Schedule& schedule,
                    structured_control_flow::Block& node, bool instrumented);

    void dispatch_node(PrettyPrinter& main_stream, PrettyPrinter& globals_stream,
                       PrettyPrinter& library_stream) override;
};

}  // namespace codegen
}  // namespace sdfg
