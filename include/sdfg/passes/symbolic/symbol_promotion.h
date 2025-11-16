#pragma once

#include "sdfg/passes/pass.h"

namespace sdfg {
namespace passes {

class SymbolPromotion : public Pass {
private:
    bool can_be_applied(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        data_flow::DataFlowGraph& dataflow
    );

    void apply(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::Sequence& sequence,
        structured_control_flow::Block& block
    );

    bool is_safe_constant_assign(
        sdfg::StructuredSDFG& sdfg, data_flow::DataFlowGraph& dataflow, data_flow::Tasklet& tasklet
    );

    bool is_safe_trunc(
        sdfg::StructuredSDFG& sdfg,
        analysis::AnalysisManager& analysis_manager,
        data_flow::DataFlowGraph& dataflow,
        data_flow::Tasklet& tasklet
    );

public:
    SymbolPromotion();

    std::string name() override;

    virtual bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    static symbolic::Expression
    as_symbol(const data_flow::DataFlowGraph& dataflow, const data_flow::Tasklet& tasklet, const std::string& op);
};

} // namespace passes
} // namespace sdfg
