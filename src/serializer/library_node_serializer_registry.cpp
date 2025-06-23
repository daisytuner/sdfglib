#include "sdfg/serializer/library_node_serializer_registry.h"

#include <string>

#include "sdfg/data_flow/library_node.h"
#include "sdfg/serializer/library_nodes/barrier_local_node_serializer.h"

namespace sdfg {
namespace serializer {

void LibraryNodeSerializerRegistry::register_library_node_serializer(std::string library_node_code,
                                                                     LibraryNodeSerializerFn fn) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (factory_map_.find(library_node_code) != factory_map_.end()) {
        throw std::runtime_error(
            "Library node serializer already registered for library node code: " +
            std::string(library_node_code));
    }
    factory_map_[library_node_code] = std::move(fn);
}

LibraryNodeSerializerFn LibraryNodeSerializerRegistry::get_library_node_serializer(
    std::string library_node_code) {
    auto it = factory_map_.find(library_node_code);
    if (it != factory_map_.end()) {
        return it->second;
    }
    return nullptr;
}

size_t LibraryNodeSerializerRegistry::size() const { return factory_map_.size(); }

void register_default_serializers() {
    LibraryNodeSerializerRegistry::instance().register_library_node_serializer(
        data_flow::BARRIER_LOCAL.value(),
        []() { return std::make_unique<serializer::BarrierLocalNodeSerializer>(); });
    // Add more serializers as needed
}

}  // namespace serializer
}  // namespace sdfg