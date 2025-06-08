#pragma once

#include <mutex>
#include <typeindex>
#include <unordered_map>

#include "sdfg/codegen/dispatchers/node_dispatcher.h"
#include "sdfg/codegen/instrumentation/instrumentation.h"
#include "sdfg/codegen/language_extension.h"

namespace sdfg {
namespace codegen {

using NodeDispatcherFn = std::function<std::unique_ptr<NodeDispatcher>(
    LanguageExtension&, StructuredSDFG&, structured_control_flow::ControlFlowNode&,
    Instrumentation&)>;

class NodeDispatcherRegistry {
   private:
    mutable std::mutex mutex_;
    std::unordered_map<std::type_index, NodeDispatcherFn> factory_map_;

   public:
    static NodeDispatcherRegistry& instance() {
        static NodeDispatcherRegistry registry;
        return registry;
    }

    void register_dispatcher(std::type_index type, NodeDispatcherFn fn) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (factory_map_.find(type) != factory_map_.end()) {
            throw std::runtime_error("Dispatcher already registered for type: " +
                                     std::string(type.name()));
        }
        factory_map_[type] = std::move(fn);
    }

    NodeDispatcherFn get_dispatcher(std::type_index type) const {
        auto it = factory_map_.find(type);
        if (it != factory_map_.end()) {
            return it->second;
        }
        return nullptr;
    }

    size_t size() const { return factory_map_.size(); }
};

std::unique_ptr<NodeDispatcher> create_dispatcher(LanguageExtension& language_extension,
                                                  StructuredSDFG& sdfg,
                                                  structured_control_flow::ControlFlowNode& node,
                                                  Instrumentation& instrumentation);

void register_default_dispatchers();

}  // namespace codegen
}  // namespace sdfg
