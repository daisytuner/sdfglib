#pragma once

#include <vector>

#include "sdfg/data_flow/library_node.h"

namespace sdfg {

namespace builder {
class SDFGBuilder;
class StructuredSDFGBuilder;
}  // namespace builder

namespace data_flow {

class DeviceDeallocNode : public LibraryNode {
    friend class sdfg::builder::SDFGBuilder;
    friend class sdfg::builder::StructuredSDFGBuilder;

   private:
    symbolic::Expression _device_id;
    std::string _container;

   protected:
    DeviceDeallocNode(const DebugInfo& debug_info, const graph::Vertex vertex,
                      DataFlowGraph& parent, const data_flow::LibraryNodeCode code,
                      const std::vector<std::string>& outputs,
                      const std::vector<std::string>& inputs, const bool side_effect = true);

   public:
    DeviceDeallocNode(const DeviceDeallocNode& data_node) = delete;
    DeviceDeallocNode& operator=(const DeviceDeallocNode&) = delete;

    virtual ~DeviceDeallocNode() = default;

    const LibraryNodeCode& code() const;

    const std::vector<std::string>& inputs() const;

    const std::vector<std::string>& outputs() const;

    const std::string& input(size_t index) const;

    const std::string& output(size_t index) const;

    const symbolic::Expression& device_id() const;

    const std::string& container() const;

    bool side_effect() const;

    bool needs_connector(size_t index) const override;

    std::unique_ptr<DataFlowNode> clone(const graph::Vertex vertex,
                                        DataFlowGraph& parent) const override;

    void replace(const symbolic::Expression& old_expression,
                 const symbolic::Expression& new_expression) override;
};

}  // namespace data_flow
}  // namespace sdfg
