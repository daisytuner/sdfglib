#pragma once

#include <sdfg/analysis/analysis.h>
#include <sdfg/structured_sdfg.h>
#include <sdfg/transformations/transformation.h>

#include <concepts>
#include <nlohmann/json.hpp>

namespace sdfg {
namespace transformations {

template<typename T>
concept transformation_concept = std::derived_from<T, sdfg::transformations::Transformation>;

class Recorder {
private:
    nlohmann::json history_;

public:
    Recorder();

    template<typename T, typename... Args>
        requires transformation_concept<T>
    void apply(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        bool skip_if_not_applicable,
        Args&&... args
    ) {
        T transformation(std::forward<Args>(args)...);

        if (!transformation.can_be_applied(builder, analysis_manager)) {
            if (!skip_if_not_applicable) {
                throw transformations::
                    InvalidTransformationException("Transformation " + transformation.name() + " cannot be applied.");
            }
            return;
        }

        nlohmann::json desc;
        transformation.to_json(desc);
        history_.push_back(desc);

        transformation.apply(builder, analysis_manager);
    };

    void save(std::filesystem::path path) const;

    nlohmann::json get_history() const { return history_; }
    nlohmann::json& history() { return history_; }
};

} // namespace transformations
} // namespace sdfg
