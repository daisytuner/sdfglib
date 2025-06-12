#pragma once

#include "sdfg/codegen/language_extension.h"
#include "sdfg/codegen/utils.h"
#include "sdfg/function.h"

namespace sdfg {
namespace codegen {

class LibraryNodeDispatcher {
   protected:
    LanguageExtension& language_extension_;
    const Function& function_;
    const data_flow::DataFlowGraph& data_flow_graph_;
    const data_flow::LibraryNode& node_;

   public:
    LibraryNodeDispatcher(LanguageExtension& language_extension, const Function& function,
                          const data_flow::DataFlowGraph& data_flow_graph,
                          const data_flow::LibraryNode& node);

    virtual ~LibraryNodeDispatcher() = default;

    virtual void dispatch(PrettyPrinter& stream) = 0;
};

}  // namespace codegen
}  // namespace sdfg