#pragma once

#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

/**
 * @brief Loop skewing transformation for nested loops
 * 
 * This transformation applies loop skewing to two nested loops, transforming
 * the iteration space to enable better parallelization or locality by changing
 * the order in which loop iterations execute.
 * 
 * Loop skewing is useful for:
 * - Exposing parallelism in loops with loop-carried dependencies
 * - Improving cache locality by changing the iteration order
 * - Enabling other transformations like wavefront parallelization
 * 
 * Theoretical transformation:
 *   for (i = lb_i; i < ub_i; i++)
 *     for (j = lb_j; j < ub_j; j++)
 *       body[i][j]
 * 
 * Becomes:
 *   for (i' = lb_i; i' < ub_i; i'++)
 *     for (j' = lb_j + skew_factor * (i' - lb_i); j' < ub_j + skew_factor * (i' - lb_i); j'++)
 *       body[i'][j' - skew_factor * (i' - lb_i)]
 * 
 * Current Implementation Note:
 * This is a basic implementation that adjusts the inner loop's lower bound but
 * does not modify the upper bound condition or memory access patterns in the body.
 * See implementation file for details on limitations and future enhancements.
 * 
 * Prerequisites:
 * - Two properly nested loops (outer loop contains only inner loop)
 * - Inner loop bounds must not depend on outer loop iteration variable
 * - At least one loop must be a Map
 * - Non-zero skew factor
 */
class LoopSkewing : public Transformation {
    structured_control_flow::StructuredLoop& outer_loop_;
    structured_control_flow::StructuredLoop& inner_loop_;
    int skew_factor_;

public:
    /**
     * @brief Construct a new Loop Skewing transformation
     * 
     * @param outer_loop The outer loop to skew
     * @param inner_loop The inner loop to skew
     * @param skew_factor The skewing factor (default: 1)
     */
    LoopSkewing(
        structured_control_flow::StructuredLoop& outer_loop,
        structured_control_flow::StructuredLoop& inner_loop,
        int skew_factor = 1
    );

    virtual std::string name() const override;

    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        override;

    virtual void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual void to_json(nlohmann::json& j) const override;

    static LoopSkewing from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);
};

} // namespace transformations
} // namespace sdfg
