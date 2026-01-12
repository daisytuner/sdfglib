#pragma once

#include "sdfg/passes/pass.h"

namespace sdfg {
namespace passes {

/**
 * @brief Pass that propagates symbolic assignments through the SDFG.
 *
 * This pass performs forward and reverse propagation of symbolic expressions
 * throughout the SDFG, replacing symbol uses with their assigned values where
 * appropriate. It handles various SDFG elements including:
 * - Transitions (symbol assignments)
 * - Memlets (data access patterns)
 * - Control flow constructs (if-else, for, while)
 * - Access nodes and library nodes
 *
 * The pass performs two main operations:
 * 1. Forward propagation: Replaces symbol reads with their assigned values
 *    when the write dominates the read and no conflicting writes occur.
 * 2. Reverse propagation: Moves symbol assignments from a join point into
 *    the branches that define the symbols.
 *
 * Only transient integer scalar symbols are considered for propagation.
 */
class SymbolPropagation : public Pass {
public:
    /**
     * @brief Constructs a new SymbolPropagation pass.
     */
    SymbolPropagation();

    /**
     * @brief Returns the name of this pass.
     * @return The string "SymbolPropagation"
     */
    virtual std::string name() override;

    /**
     * @brief Executes the symbol propagation pass on the SDFG.
     *
     * @param builder The structured SDFG builder containing the SDFG to optimize
     * @param analysis_manager The analysis manager providing required analyses
     * @return true if the pass modified the SDFG, false otherwise
     */
    virtual bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
};

} // namespace passes
} // namespace sdfg
