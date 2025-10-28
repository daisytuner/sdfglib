#pragma once

#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/passes/pass.h"

namespace sdfg {
namespace passes {

class ReferencePropagation : public Pass {
private:
    void merge_access_nodes(builder::StructuredSDFGBuilder& builder, data_flow::AccessNode& user_node);

public:
    ReferencePropagation();

    std::string name() override;

    virtual bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
};

} // namespace passes
} // namespace sdfg
