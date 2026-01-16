#pragma once

#include "sdfg/structured_control_flow/map.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

class GPULoopReordering : public Transformation {
private:
    structured_control_flow::Map& map_;

public:
    bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    GPULoopReordering(structured_control_flow::Map& map_);

    virtual std::string name() const override;

    virtual void to_json(nlohmann::json& j) const override;

    static GPULoopReordering from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);
};

} // namespace transformations
} // namespace sdfg
