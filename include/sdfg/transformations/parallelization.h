#pragma once

#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

class Parallelization : public Transformation {
    structured_control_flow::Map& map_;

   public:
    Parallelization(structured_control_flow::Map& map);

    virtual std::string name() const override;

    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder,
                                analysis::AnalysisManager& analysis_manager) override;

    virtual void apply(builder::StructuredSDFGBuilder& builder,
                       analysis::AnalysisManager& analysis_manager) override;

    virtual void to_json(nlohmann::json& j) const override;

    static Parallelization from_json(builder::StructuredSDFGBuilder& builder,
                                     const nlohmann::json& j);
};

}  // namespace transformations
}  // namespace sdfg
