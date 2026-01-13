#pragma once

#include "sdfg/analysis/scop_analysis.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

class ISLScheduler : public Transformation {
    structured_control_flow::StructuredLoop& loop_;

    std::unique_ptr<analysis::Scop> scop_;

    std::unique_ptr<analysis::Dependences> dependences_;

    bool applied_ = false;

public:
    ISLScheduler(structured_control_flow::StructuredLoop& loop);

    virtual std::string name() const override;

    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        override;

    virtual void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual void to_json(nlohmann::json& j) const override;

    static ISLScheduler from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);
};

} // namespace transformations
} // namespace sdfg
