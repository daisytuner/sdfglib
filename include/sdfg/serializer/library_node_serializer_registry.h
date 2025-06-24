#pragma once

#include <functional>
#include <memory>
#include <nlohmann/json_fwd.hpp>

#include "sdfg/serializer/json_serializer.h"

namespace sdfg {
namespace serializer {

using LibraryNodeSerializerFn = std::function<std::unique_ptr<LibraryNodeSerializer>()>;

class LibraryNodeSerializerRegistry {
   private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, LibraryNodeSerializerFn> factory_map_;

   public:
    static LibraryNodeSerializerRegistry& instance() {
        static LibraryNodeSerializerRegistry registry;
        return registry;
    }

    void register_library_node_serializer(std::string library_node_code,
                                          LibraryNodeSerializerFn fn);

    LibraryNodeSerializerFn get_library_node_serializer(std::string library_node_code);

    size_t size() const;
};

void register_default_serializers();

}  // namespace serializer
}  // namespace sdfg
