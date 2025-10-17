#pragma once

#include "sdfg/passes/pass.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace passes {

class DOKScheduling : public Pass {
private:
    symbolic::Expression load_threshold;

    int balance_threshold;

    symbolic::Expression size_threshold;

    symbolic::Expression number_threshold;

    int avail_threads;

    void read_thresholds();

public:
    DOKScheduling();

    std::string name() override;

    virtual bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
};

} // namespace passes
} // namespace sdfg
