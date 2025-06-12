#pragma once

#include "sdfg/codegen/dispatchers/node_dispatcher.h"
#include "sdfg/structured_control_flow/map.h"

namespace sdfg {
namespace codegen {

class MapDispatcher : public NodeDispatcher {
   private:
    structured_control_flow::Map& node_;

   public:
    MapDispatcher(LanguageExtension& language_extension, StructuredSDFG& sdfg,
                  structured_control_flow::Map& node, Instrumentation& instrumentation);

    void dispatch_node(PrettyPrinter& main_stream, PrettyPrinter& globals_stream,
                       PrettyPrinter& library_stream) override;
};

class SequentialMapDispatcher : public NodeDispatcher {
   private:
    structured_control_flow::Map& node_;

   public:
    SequentialMapDispatcher(LanguageExtension& language_extension, StructuredSDFG& sdfg,
                            structured_control_flow::Map& node, Instrumentation& instrumentation);

    void dispatch_node(PrettyPrinter& main_stream, PrettyPrinter& globals_stream,
                       PrettyPrinter& library_stream) override;
};

class CPUParallelMapDispatcher : public NodeDispatcher {
   private:
    structured_control_flow::Map& node_;

   public:
    CPUParallelMapDispatcher(LanguageExtension& language_extension, StructuredSDFG& sdfg,
                             structured_control_flow::Map& node, Instrumentation& instrumentation);

    void dispatch_node(PrettyPrinter& main_stream, PrettyPrinter& globals_stream,
                       PrettyPrinter& library_stream) override;
};

using MapDispatcherFn = std::function<std::unique_ptr<NodeDispatcher>(
    LanguageExtension&, StructuredSDFG&, structured_control_flow::Map&, Instrumentation&)>;

class MapDispatcherRegistry {
   private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string_view, MapDispatcherFn> factory_map_;

   public:
    static MapDispatcherRegistry& instance() {
        static MapDispatcherRegistry registry;
        return registry;
    }

    void register_map_dispatcher(std::string_view schedule_type, MapDispatcherFn fn) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (factory_map_.find(schedule_type) != factory_map_.end()) {
            throw std::runtime_error("Map dispatcher already registered for schedule type: " +
                                     std::string(schedule_type));
        }
        factory_map_[schedule_type] = std::move(fn);
    }

    MapDispatcherFn get_map_dispatcher(std::string_view schedule_type) const {
        auto it = factory_map_.find(schedule_type);
        if (it != factory_map_.end()) {
            return it->second;
        }
        return nullptr;
    }

    size_t size() const { return factory_map_.size(); }
};

void register_default_map_dispatchers();

}  // namespace codegen
}  // namespace sdfg
