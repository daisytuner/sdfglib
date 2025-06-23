#include "sdfg/serializer/library_nodes/barrier_local_node_serializer.h"

#include "sdfg/data_flow/library_nodes/barrier_local_node.h"

namespace sdfg {
namespace serializer {

nlohmann::json BarrierLocalNodeSerializer::serialize(
    const sdfg::data_flow::LibraryNode& library_node) {
    if (library_node.code() != data_flow::BARRIER_LOCAL) {
        throw std::runtime_error("Invalid library node code");
    }
    nlohmann::json j;
    j["code"] = std::string(library_node.code().value());
    return j;
}

data_flow::LibraryNode& BarrierLocalNodeSerializer::deserialize(
    const nlohmann::json& j, sdfg::builder::StructuredSDFGBuilder& builder,
    sdfg::structured_control_flow::Block& parent) {
    auto code = j["code"].get<std::string>();
    if (code != data_flow::BARRIER_LOCAL.value()) {
        throw std::runtime_error("Invalid library node code");
    }
    return builder.add_library_node<data_flow::BarrierLocalNode>(parent, data_flow::BARRIER_LOCAL,
                                                                 {}, {}, false);
};

}  // namespace serializer
}  // namespace sdfg