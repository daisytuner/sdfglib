#pragma once

#include "sdfg/data_flow/library_node.h"
#include "sdfg/structured_control_flow/map.h"

namespace sdfg {

namespace analysis {
class AnalysisManager;
}

namespace builder {
class StructuredSDFGBuilder;
}

namespace math {


class MathNode : public data_flow::LibraryNode {
public:
    MathNode(
        size_t element_id,
        const DebugInfoRegion& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const data_flow::LibraryNodeCode& code,
        const std::vector<std::string>& outputs,
        const std::vector<std::string>& inputs,
        const data_flow::ImplementationType& implementation_type
    );

    virtual bool expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) = 0;

    symbolic::SymbolSet symbols() const override { return {}; }

    void replace(const symbolic::Expression& old_expression, const symbolic::Expression& new_expression) override {
        return;
    }
};

} // namespace math
} // namespace sdfg
