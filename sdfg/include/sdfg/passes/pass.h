#pragma once

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/sdfg_builder.h"
#include "sdfg/builder/structured_sdfg_builder.h"

namespace sdfg {
namespace passes {

class Pass {
public:
    virtual ~Pass() = default;

    virtual std::string name() = 0;

    bool run(builder::SDFGBuilder& builder, bool create_report = false);

    bool
    run(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager, bool create_report = false
    );

    virtual bool run_pass(builder::SDFGBuilder& builder);

    virtual bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    virtual void invalidates(analysis::AnalysisManager& analysis_manager, bool applied);
};

template<typename T>
class VisitorPass : public Pass {
    std::string name() override { return T::name(); };

    bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override {
        T visitor(builder, analysis_manager);
        return visitor.visit();
    };
};

} // namespace passes
} // namespace sdfg
