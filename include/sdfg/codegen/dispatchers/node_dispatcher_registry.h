#pragma once

#include <mutex>
#include <typeindex>
#include <unordered_map>

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/codegen/dispatchers/node_dispatcher.h"
#include "sdfg/codegen/instrumentation/instrumentation_plan.h"
#include "sdfg/codegen/language_extension.h"

namespace sdfg {
namespace codegen {

using NodeDispatcherFn = std::function<std::unique_ptr<
    NodeDispatcher>(LanguageExtension&, StructuredSDFG&, structured_control_flow::ControlFlowNode&, InstrumentationPlan&)>;

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
            throw std::runtime_error("Dispatcher already registered for type: " + std::string(type.name()));
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

std::unique_ptr<NodeDispatcher> create_dispatcher(
    LanguageExtension& language_extension,
    StructuredSDFG& sdfg,
    structured_control_flow::ControlFlowNode& node,
    InstrumentationPlan& instrumentation_plan
);

using MapDispatcherFn = std::function<std::unique_ptr<
    NodeDispatcher>(LanguageExtension&, StructuredSDFG&, structured_control_flow::Map&, InstrumentationPlan&)>;

class MapDispatcherRegistry {
private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, MapDispatcherFn> factory_map_;

public:
    static MapDispatcherRegistry& instance() {
        static MapDispatcherRegistry registry;
        return registry;
    }

    void register_map_dispatcher(std::string schedule_type, MapDispatcherFn fn) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (factory_map_.find(schedule_type) != factory_map_.end()) {
            throw std::runtime_error("Map dispatcher already registered for schedule type: " + std::string(schedule_type));
        }
        factory_map_[schedule_type] = std::move(fn);
    }

    MapDispatcherFn get_map_dispatcher(std::string schedule_type) const {
        auto it = factory_map_.find(schedule_type);
        if (it != factory_map_.end()) {
            return it->second;
        }
        return nullptr;
    }

    size_t size() const { return factory_map_.size(); }
};

using LibraryNodeDispatcherFn = std::function<std::unique_ptr<
    LibraryNodeDispatcher>(LanguageExtension&, const Function&, const data_flow::DataFlowGraph&, const data_flow::LibraryNode&)>;

class LibraryNodeDispatcherRegistry {
private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, LibraryNodeDispatcherFn> factory_map_;

public:
    static LibraryNodeDispatcherRegistry& instance() {
        static LibraryNodeDispatcherRegistry registry;
        return registry;
    }

    void register_library_node_dispatcher(std::string library_node_code, LibraryNodeDispatcherFn fn) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (factory_map_.find(library_node_code) != factory_map_.end()) {
            throw std::runtime_error(
                "Library node dispatcher already registered for library node code: " + std::string(library_node_code)
            );
        }
        factory_map_[library_node_code] = std::move(fn);
    }

    LibraryNodeDispatcherFn get_library_node_dispatcher(std::string library_node_code) const {
        auto it = factory_map_.find(library_node_code);
        if (it != factory_map_.end()) {
            return it->second;
        }
        return nullptr;
    }

    size_t size() const { return factory_map_.size(); }
};

void register_default_dispatchers();

} // namespace codegen
} // namespace sdfg
