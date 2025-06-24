#pragma once

#include <sdfg/analysis/analysis.h>
#include <sdfg/structured_sdfg.h>
#include <sdfg/transformations/transformation.h>

#include <concepts>
#include <nlohmann/json.hpp>

namespace sdfg {
namespace transformations {

template <typename T>
concept transformation_concept = std::derived_from<T, sdfg::transformations::Transformation>;

class Recorder {
   private:
    nlohmann::json history_;

   public:
    Recorder();

    template <typename T, typename... Args>
        requires transformation_concept<T>
    void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager,
               Args&&... args) {
        T transformation(std::forward<Args>(args)...);

        if (!transformation.can_be_applied(builder, analysis_manager)) {
            return;
        }

        nlohmann::json desc;
        transformation.to_json(desc);
        std::cout << desc.dump(4) << std::endl;
        history_.push_back(desc);

        transformation.apply(builder, analysis_manager);
    };

    void replay(builder::StructuredSDFGBuilder& builder,
                analysis::AnalysisManager& analysis_manager, const nlohmann::json& desc);

    template <typename T>
        requires transformation_concept<T>
    void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager,
               const nlohmann::json& desc) {
        T transformation(T::from_json(builder, desc));
        if (!transformation.can_be_applied(builder, analysis_manager)) {
            std::cout << "Cannot apply transformation: " << transformation.name() << std::endl;
            return;
        }
        transformation.apply(builder, analysis_manager);
    };

    void save(std::filesystem::path path) const;
};

}  // namespace transformations
}  // namespace sdfg
