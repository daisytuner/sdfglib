/**
 * @file memlet_base_type_normalization.h
 * @brief Pass for normalizing memlet base types
 *
 * This pass converts pointers to nested array types into flat pointers with
 * the element type of the innermost array. The subset is flattened into a
 * linearized access using the num_elements property of the arrays.
 */

#pragma once

#include <string>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/passes/pass.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

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
class MemletBaseTypeNormalization : public visitor::NonStoppingStructuredSDFGVisitor {
public:
    /**
     * @brief Constructs a new MemletBaseTypeNormalization pass
     * @param builder The structured SDFG builder
     * @param analysis_manager The analysis manager
     */
    MemletBaseTypeNormalization(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    /**
     * @brief Returns the name of the pass
     * @return The string "MemletBaseTypeNormalization"
     */
    static std::string name() { return "MemletBaseTypeNormalization"; }

    /**
     * @brief Accepts a block and normalizes memlets in its dataflow graph
     * @param block The block to process
     * @return true if any memlets were normalized
     */
    virtual bool accept(structured_control_flow::Block& block) override;
};

typedef VisitorPass<MemletBaseTypeNormalization> MemletBaseTypeNormalizationPass;

} // namespace passes
} // namespace sdfg
