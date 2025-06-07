#pragma once

#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

class OutLocalStorage : public Transformation {
    structured_control_flow::Sequence& parent_;
    structured_control_flow::StructuredLoop& loop_;
    std::string container_;
    bool requires_array_;

   public:
    OutLocalStorage(structured_control_flow::Sequence& parent,
                    structured_control_flow::StructuredLoop& loop, std::string container);

    virtual std::string name() override;

    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder,
                                analysis::AnalysisManager& analysis_manager) override;

    virtual void apply(builder::StructuredSDFGBuilder& builder,
                       analysis::AnalysisManager& analysis_manager) override;

   private:
    void apply_array(builder::StructuredSDFGBuilder& builder,
                     analysis::AnalysisManager& analysis_manager);

    void apply_scalar(builder::StructuredSDFGBuilder& builder,
                      analysis::AnalysisManager& analysis_manager);
};

}  // namespace transformations
}  // namespace sdfg
