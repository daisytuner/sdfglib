/**
 * @file memlet_base_type_normalization.h
 * @brief Pass for normalizing memlet base types
 *
 * This pass converts pointers to nested array types into flat pointers with
 * the element type of the innermost array. The subset is flattened into a
 * linearized access using the num_elements property of the arrays.
 */

#pragma once

#include "sdfg/passes/pass.h"

namespace sdfg {
namespace passes {

/**
 * @class MemletBaseTypeNormalization
 * @brief Normalizes memlet base types by flattening nested arrays
 *
 * This pass transforms memlets that have pointer base types pointing to nested
 * arrays. It converts such pointers into flat pointers to the innermost element
 * type, and adjusts the subset accordingly by linearizing multi-dimensional
 * array accesses into a single linear index.
 *
 * For example, a pointer to int[3][4] with subset [i, j] becomes a pointer to
 * int with subset [i * 4 + j].
 *
 * This normalization simplifies subsequent passes and code generation by
 * eliminating nested array types in pointer pointees.
 */
class MemletBaseTypeNormalization : public Pass {
public:
    /**
     * @brief Constructs a new MemletBaseTypeNormalization pass
     */
    MemletBaseTypeNormalization();

    /**
     * @brief Returns the name of the pass
     * @return The string "MemletBaseTypeNormalization"
     */
    virtual std::string name() override;

    /**
     * @brief Runs the pass on the given SDFG builder
     * @param builder The structured SDFG builder
     * @param analysis_manager The analysis manager
     * @return true if the pass made changes, false otherwise
     */
    virtual bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
};

} // namespace passes
} // namespace sdfg
