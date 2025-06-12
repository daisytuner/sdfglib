#pragma once

#include "sdfg/codegen/dispatchers/library_nodes/library_node_dispatcher.h"
#include "sdfg/data_flow/barrier_local_node.h"

namespace sdfg {
namespace codegen {

class ThreadBarrierDispatcher : public LibraryNodeDispatcher {
   public:
    ThreadBarrierDispatcher(LanguageExtension& language_extension, const Function& function,
                            const data_flow::DataFlowGraph& data_flow_graph,
                            const data_flow::BarrierLocalNode& node)
        : LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

    void dispatch(PrettyPrinter& stream) override;
};

}  // namespace codegen
}  // namespace sdfg