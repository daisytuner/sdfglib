#pragma once

#include "sdfg/codegen/dispatchers/node_dispatcher_registry.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/utils.h"

namespace sdfg {
namespace codegen {

class BlockDispatcher : public NodeDispatcher {
   private:
    const structured_control_flow::Block& node_;

   public:
    BlockDispatcher(LanguageExtension& language_extension, StructuredSDFG& sdfg,
                    structured_control_flow::Block& node, Instrumentation& instrumentation);

    void dispatch_node(PrettyPrinter& main_stream, PrettyPrinter& globals_stream,
                       PrettyPrinter& library_stream) override;
};

class DataFlowDispatcher {
   private:
    LanguageExtension& language_extension_;
    const Function& function_;
    const data_flow::DataFlowGraph& data_flow_graph_;

    void dispatch_ref(PrettyPrinter& stream, const data_flow::Memlet& memlet);

    void dispatch_tasklet(PrettyPrinter& stream, const data_flow::Tasklet& tasklet);

    void dispatch_library_node(PrettyPrinter& stream, const data_flow::LibraryNode& libnode);

   public:
    DataFlowDispatcher(LanguageExtension& language_extension, const Function& function,
                       const data_flow::DataFlowGraph& data_flow_graph);

    void dispatch(PrettyPrinter& stream);
};

class LibraryNodeDispatcher {
   private:
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

using LibraryNodeDispatcherFn = std::function<std::unique_ptr<LibraryNodeDispatcher>(
    LanguageExtension&, const Function&, const data_flow::DataFlowGraph&,
    const data_flow::LibraryNode&)>;

class LibraryNodeDispatcherRegistry {
   private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string_view, LibraryNodeDispatcherFn> factory_map_;

   public:
    static LibraryNodeDispatcherRegistry& instance() {
        static LibraryNodeDispatcherRegistry registry;
        return registry;
    }

    void register_library_node_dispatcher(std::string_view library_node_code,
                                          LibraryNodeDispatcherFn fn) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (factory_map_.find(library_node_code) != factory_map_.end()) {
            throw std::runtime_error(
                "Library node dispatcher already registered for library node code: " +
                std::string(library_node_code));
        }
        factory_map_[library_node_code] = std::move(fn);
    }

    LibraryNodeDispatcherFn get_library_node_dispatcher(std::string_view library_node_code) const {
        auto it = factory_map_.find(library_node_code);
        if (it != factory_map_.end()) {
            return it->second;
        }
        return nullptr;
    }

    size_t size() const { return factory_map_.size(); }
};

inline void register_default_library_node_dispatchers() {};

}  // namespace codegen
}  // namespace sdfg
