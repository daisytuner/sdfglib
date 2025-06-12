#pragma once

#include <nlohmann/json_fwd.hpp>

#include "sdfg/data_flow/library_node.h"
#include "sdfg/serializer/json_serializer.h"

namespace sdfg {
namespace serializer {
inline constexpr data_flow::LibraryNodeCode BARRIER_LOCAL{"barrier_local"};
class BarrierLocalNodeSerializer : public serializer::LibraryNodeSerializer {
   public:
    nlohmann::json serialize(const sdfg::data_flow::LibraryNode& library_node) override;

    data_flow::LibraryNode& deserialize(const nlohmann::json& j,
                                        sdfg::builder::StructuredSDFGBuilder& builder,
                                        sdfg::structured_control_flow::Block& parent) override;
};

}  // namespace serializer
}  // namespace sdfg
