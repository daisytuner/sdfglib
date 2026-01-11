#pragma once

#include "sdfg/passes/pass.h"

namespace sdfg {
namespace passes {

/**
 * @brief Optimization pass that performs scalar evolution analysis on loop-dependent symbols.
 *
 * The SymbolEvolution pass analyzes scalar variables that are updated within loops and attempts
 * to derive closed-form expressions for their values based on the loop induction variable.
 * This enables optimizations such as loop-invariant code motion and symbolic simplification.
 *
 * The pass recognizes several common evolution patterns:
 * - Pattern 1: Constant values that don't change
 * - Pattern 2: Loop aliases that directly track the induction variable
 * - Pattern 3: Functions of the previous iteration's induction variable
 * - Pattern 4: Affine updates (increments/decrements by a constant each iteration)
 *
 * For each pattern, the pass:
 * 1. Verifies the symbol meets specific criteria (single write, not in nested loop, etc.)
 * 2. Derives a closed-form expression for the symbol's evolution
 * 3. Redefines the symbol at the loop entry with the derived expression
 * 4. Updates the transition after the loop to set the final value
 */
class SymbolEvolution : public Pass {
private:
    /**
     * @brief Attempts to eliminate loop-dependent symbols by deriving their evolution.
     *
     * @param builder The SDFG builder for modifying the graph
     * @param analysis_manager The analysis manager providing dataflow and dominance information
     * @param loop The structured loop to analyze
     * @param transition The transition after the loop (for setting final values)
     * @return true if any symbols were successfully eliminated, false otherwise
     */
    bool eliminate_symbols(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::StructuredLoop& loop,
        structured_control_flow::Transition& transition
    );

public:
    /**
     * @brief Constructs a new SymbolEvolution pass instance.
     */
    SymbolEvolution();

    /**
     * @brief Returns the name of this pass.
     *
     * @return "SymbolEvolution"
     */
    std::string name() override;

    /**
     * @brief Runs the symbol evolution pass on the given SDFG.
     *
     * Traverses all structured loops in the SDFG and attempts to derive closed-form
     * expressions for loop-dependent scalar symbols.
     *
     * @param builder The SDFG builder providing access to the graph
     * @param analysis_manager The analysis manager for running required analyses
     * @return true if the pass modified the SDFG, false otherwise
     */
    virtual bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
};

} // namespace passes
} // namespace sdfg
