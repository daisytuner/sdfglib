#pragma once

#include <sdfg/analysis/analysis.h>
#include <sdfg/structured_sdfg.h>
#include <sdfg/transformations/recorder.h>
#include <sdfg/transformations/transformation.h>

#include <nlohmann/json.hpp>

namespace sdfg {
namespace transformations {

class Replayer {
public:
    void replay(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        const nlohmann::json& desc,
        bool skip_if_not_applicable = true,
        size_t loopnest_index = 0
    );

    template<typename T>
        requires transformation_concept<T>
    void apply(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        const nlohmann::json& desc,
        bool skip_if_not_applicable = true
    ) {
        T transformation(T::from_json(builder, desc));
        if (!transformation.can_be_applied(builder, analysis_manager)) {
            if (!skip_if_not_applicable) {
                throw transformations::
                    InvalidTransformationException("Transformation " + transformation.name() + " cannot be applied.");
            }
            return;
        }
        transformation.apply(builder, analysis_manager);
    };
};

} // namespace transformations
} // namespace sdfg
