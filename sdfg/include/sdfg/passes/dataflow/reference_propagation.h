#pragma once

#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/passes/pass.h"

namespace sdfg {
namespace passes {

class ReferencePropagation : public Pass {
public:
    ReferencePropagation();

    std::string name() override;

    virtual bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    static bool
    compatible_type(const Function& function, const data_flow::Memlet& reference, const data_flow::Memlet& target);
};

} // namespace passes
} // namespace sdfg
