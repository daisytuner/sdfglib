#pragma once

#include <boost/lexical_cast.hpp>
#include <nlohmann/json.hpp>

#include "sdfg/element.h"
#include "sdfg/graph/graph.h"

using json = nlohmann::json;

namespace sdfg {

namespace builder {
class SDFGBuilder;
class StructuredSDFGBuilder;
}  // namespace builder

namespace data_flow {

class DataFlowGraph;

class DataFlowNode : public Element {
    friend class sdfg::builder::SDFGBuilder;
    friend class sdfg::builder::StructuredSDFGBuilder;

   private:
    // Remark: Exclusive resource
    graph::Vertex vertex_;

    DataFlowGraph* parent_;

   protected:
    DataFlowNode(const DebugInfo& debug_info, const graph::Vertex vertex, DataFlowGraph& parent);

   public:
    // Remark: Exclusive resource
    DataFlowNode(const DataFlowNode& data_node) = delete;
    DataFlowNode& operator=(const DataFlowNode&) = delete;

    graph::Vertex vertex() const;

    const DataFlowGraph& get_parent() const;

    DataFlowGraph& get_parent();

    virtual std::unique_ptr<DataFlowNode> clone(const graph::Vertex vertex,
                                                DataFlowGraph& parent) const = 0;
};
}  // namespace data_flow
}  // namespace sdfg