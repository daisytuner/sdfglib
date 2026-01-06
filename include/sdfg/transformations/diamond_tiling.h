#pragma once

#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

/**
 * @brief Diamond tiling transformation for n-dimensional loop nests
 * 
 * Diamond tiling is a loop transformation that combines tiling and loop interchange
 * to improve data locality and enable parallelization for stencil computations.
 * This transformation creates a "diamond" pattern in the iteration space by:
 * 1. Tiling the outer loop with the specified outer tile size
 * 2. Tiling the inner loop with the specified inner tile size
 * 3. Interchanging the original outer loop with the tiled inner loop
 * 
 * The result is a loop structure that improves temporal locality by reusing data
 * within tiles across multiple time steps (when applied to time-space loop nests).
 * 
 * This implementation is kept simple by reusing existing LoopTiling and LoopInterchange
 * transformations rather than implementing a custom diamond tiling algorithm.
 * 
 * @example
 * Original 2D loop nest:
 * @code
 * for i = 0 to N:
 *   for j = 0 to M:
 *     A[i][j] = ...
 * @endcode
 * 
 * After diamond tiling (outer_tile_size=32, inner_tile_size=8):
 * @code
 * for i_tile = 0 to N step 32:
 *   for j_tile = 0 to M step 8:
 *     for i = i_tile to min(i_tile+32, N):
 *       for j = j_tile to min(j_tile+8, M):
 *         A[i][j] = ...
 * @endcode
 */
class DiamondTiling : public Transformation {
    structured_control_flow::StructuredLoop& outer_loop_;
    structured_control_flow::StructuredLoop& inner_loop_;
    size_t outer_tile_size_;
    size_t inner_tile_size_;

public:
    /**
     * @brief Construct a diamond tiling transformation
     * 
     * @param outer_loop The outermost loop in the 2D loop nest (typically time dimension)
     * @param inner_loop The inner loop in the 2D loop nest (typically space dimension)
     * @param outer_tile_size Tile size for the outer loop (must be > 1)
     * @param inner_tile_size Tile size for the inner loop (must be > 1)
     */
    DiamondTiling(
        structured_control_flow::StructuredLoop& outer_loop,
        structured_control_flow::StructuredLoop& inner_loop,
        size_t outer_tile_size,
        size_t inner_tile_size
    );

    /**
     * @brief Get the name of this transformation
     * @return "DiamondTiling"
     */
    virtual std::string name() const override;

    /**
     * @brief Check if diamond tiling can be applied to the loop nest
     * 
     * Diamond tiling can be applied if:
     * 1. Both tile sizes are greater than 1
     * 2. The inner loop is the only child of the outer loop's body
     * 3. The inner loop does not depend on the outer loop's induction variable
     * 4. Both loops are contiguous (unit stride)
     * 
     * @param builder The SDFG builder
     * @param analysis_manager Analysis manager for dependency checks
     * @return true if the transformation can be applied, false otherwise
     */
    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        override;

    /**
     * @brief Apply diamond tiling to the loop nest
     * 
     * The transformation proceeds in three steps:
     * 1. Tile the outer loop with outer_tile_size
     * 2. Tile the inner loop with inner_tile_size
     * 3. Interchange the original outer loop with the tiled inner loop
     * 
     * @param builder The SDFG builder for modifying the loop structure
     * @param analysis_manager Analysis manager that will be invalidated after transformation
     */
    virtual void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    /**
     * @brief Serialize transformation to JSON
     * 
     * @param j JSON object to write transformation description
     */
    virtual void to_json(nlohmann::json& j) const override;

    /**
     * @brief Deserialize transformation from JSON
     * 
     * @param builder The SDFG builder containing the loops to transform
     * @param j JSON object with transformation description
     * @return DiamondTiling transformation reconstructed from JSON
     */
    static DiamondTiling from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);
};

} // namespace transformations
} // namespace sdfg
