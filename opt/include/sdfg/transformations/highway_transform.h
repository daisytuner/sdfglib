#pragma once

#include "sdfg/targets/highway/schedule.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

/**
 * @brief HighwayTransform transformation for Map nodes
 *
 * This transformation changes the schedule type of a Map from sequential
 * to vectorized (HIGHWAY). This enables the loop iterations
 * to be vectorized on a range of platforms.
 *
 * @note Only applicable to Maps with sequential schedule type
 */
class HighwayTransform : public Transformation {
    structured_control_flow::Map& map_;

public:
    /**
     * @brief Construct a HighwayTransform transformation
     * @param map The map to be vectorized
     */
    HighwayTransform(structured_control_flow::Map& map);

    /**
     * @brief Get the name of this transformation
     * @return "HighwayTransform"
     */
    virtual std::string name() const override;

    /**
     * @brief Check if this transformation can be applied
     * @param builder The SDFG builder
     * @param analysis_manager The analysis manager
     * @return true if the map is sequential, memory accesses are aligned, and the tasklets are suitable for
     * vectorization
     */
    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        override;

    /**
     * @brief Apply the HighwayTransform transformation
     * @param builder The SDFG builder
     * @param analysis_manager The analysis manager
     */
    virtual void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    /**
     * @brief Serialize this transformation to JSON
     * @param j JSON object to populate
     */
    virtual void to_json(nlohmann::json& j) const override;

    /**
     * @brief Deserialize a HighwayTransform transformation from JSON
     * @param builder The SDFG builder
     * @param j JSON description of the transformation
     * @return The deserialized transformation
     */
    static HighwayTransform from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);

    enum MemletAccessType { UNKNOWN, CONSTANT, CONTIGUOUS };

    static MemletAccessType classify_memlet_access_type(
        const data_flow::Subset& subset, const symbolic::Symbol& indvar, const symbolic::SymbolSet& local_symbols
    );
};

} // namespace transformations
} // namespace sdfg
