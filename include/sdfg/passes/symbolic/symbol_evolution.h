#pragma once

#include "sdfg/passes/pass.h"

namespace sdfg {
namespace passes {

class SymbolEvolution : public Pass {
private:
    bool eliminate_symbols(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::StructuredLoop& loop,
        structured_control_flow::Transition& transition
    );

public:
    SymbolEvolution();

    std::string name() override;

    virtual bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
};

} // namespace passes
} // namespace sdfg
