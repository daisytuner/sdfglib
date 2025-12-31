#pragma once

#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

class LoopTiling : public Transformation {
    structured_control_flow::StructuredLoop& loop_;
    size_t tile_size_;

public:
    LoopTiling(structured_control_flow::StructuredLoop& loop, size_t tile_size);

    virtual std::string name() const override;

    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        override;

    virtual void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual void to_json(nlohmann::json& j) const override;

    static LoopTiling from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);
};

} // namespace transformations
} // namespace sdfg
