#pragma once

#include <vector>

#include "sdfg/data_flow/library_node.h"

namespace sdfg {

namespace builder {
class SDFGBuilder;
class StructuredSDFGBuilder;
}  // namespace builder

namespace data_flow {

class ThreadBarrierNode : public LibraryNode {
    friend class sdfg::builder::SDFGBuilder;
    friend class sdfg::builder::StructuredSDFGBuilder;

   protected:
    ThreadBarrierNode(const DebugInfo& debug_info, const graph::Vertex vertex,
                      DataFlowGraph& parent);

   public:
    ThreadBarrierNode(const ThreadBarrierNode& data_node) = delete;
    ThreadBarrierNode& operator=(const ThreadBarrierNode&) = delete;

    virtual ~ThreadBarrierNode() = default;

    const LibraryNodeCode& code() const;

    const std::vector<std::string>& inputs() const;

    const std::vector<std::string>& outputs() const;

    const std::string& input(size_t index) const;

    const std::string& output(size_t index) const;

    bool side_effect() const;

    bool needs_connector(size_t index) const override;
};

}  // namespace data_flow
}  // namespace sdfg
