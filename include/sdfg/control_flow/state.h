#pragma once

#include <boost/lexical_cast.hpp>
#include <nlohmann/json.hpp>
#include <string>

#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/element.h"
#include "sdfg/graph/graph.h"

using json = nlohmann::json;

namespace sdfg {

namespace builder {
class SDFGBuilder;
}

namespace control_flow {

class State : public Element {
    friend class sdfg::builder::SDFGBuilder;

private:
    // Remark: Exclusive resource
    const graph::Vertex vertex_;
    std::unique_ptr<data_flow::DataFlowGraph> dataflow_;

    State(size_t element_id, const DebugInfo& debug_info, const graph::Vertex vertex);

public:
    // Remark: Exclusive resource
    State(const State& state) = delete;
    State& operator=(const State&) = delete;

    void validate() const override;

    graph::Vertex vertex() const;

    const data_flow::DataFlowGraph& dataflow() const;

    data_flow::DataFlowGraph& dataflow();

    void replace(const symbolic::Expression& old_expression, const symbolic::Expression& new_expression) override;
};

} // namespace control_flow
} // namespace sdfg
