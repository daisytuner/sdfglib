/**
 * @file symbol_promotion.h
 * @brief Symbol promotion pass for converting dataflow operations to symbolic expressions
 *
 * This file defines the SymbolPromotion pass, which transforms simple computational
 * dataflow operations into symbolic expressions for improved optimization and analysis.
 */

#pragma once

#include "sdfg/passes/pass.h"

namespace sdfg {
namespace passes {

/**
 * @class SymbolPromotion
 * @brief Pass that promotes simple dataflow computations to symbolic expressions
 *
 * The SymbolPromotion pass identifies single-tasklet dataflow graphs that perform
 * simple integer arithmetic operations and converts them into symbolic expressions
 * represented as state transitions. This enables better symbolic analysis and
 * optimization of control flow.
 *
 * The pass applies to tasklets that meet the following criteria:
 * - Single tasklet with computational memlets only
 * - No floating-point operations
 * - Signed integer operands (with special cases for unsigned constants)
 * - Supported operations: assign, add, sub, mul, sdiv, srem, smin, smax, abs, shift, logical ops
 *
 * @note The pass handles special cases like zero-extension and truncation for specific type conversions
 */
class SymbolPromotion : public Pass {
private:
    /**
     * @brief Checks if the pass can be applied to a dataflow graph
     *
     * Determines whether the given dataflow graph meets all criteria for symbol promotion:
     * - Contains exactly one tasklet
     * - All edges are computational memlets
     * - Tasklet operation is not floating-point
     * - Tasklet operation is signed (unless special case)
     * - Inputs and outputs are signed integers without casts
     * - Operation is in the supported set
     *
     * @param builder The SDFG builder
     * @param analysis_manager The analysis manager
     * @param dataflow The dataflow graph to check
     * @return true if the pass can be applied, false otherwise
     */
    bool can_be_applied(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        data_flow::DataFlowGraph& dataflow
    );

    /**
     * @brief Applies the symbol promotion transformation
     *
     * Converts the tasklet operation into a symbolic expression and creates
     * a state transition with the symbolic assignment. The tasklet is then
     * removed from the dataflow graph.
     *
     * @param builder The SDFG builder
     * @param analysis_manager The analysis manager
     * @param sequence The parent sequence
     * @param block The block containing the tasklet to promote
     */
    void apply(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::Sequence& sequence,
        structured_control_flow::Block& block
    );

    /**
     * @brief Checks if an assign operation from unsigned constant to signed variable is safe
     *
     * Verifies that assigning an unsigned constant to a signed variable can be safely
     * promoted to a symbolic expression by checking:
     * - Input is an unsigned integer constant
     * - Output is a signed integer variable
     * - Constant value fits in the signed output type
     *
     * @param sdfg The structured SDFG
     * @param dataflow The dataflow graph
     * @param tasklet The assignment tasklet to check
     * @return true if the constant assignment is safe to promote, false otherwise
     */
    bool is_safe_constant_assign(
        sdfg::StructuredSDFG& sdfg, data_flow::DataFlowGraph& dataflow, data_flow::Tasklet& tasklet
    );

public:
    /**
     * @brief Default constructor
     */
    SymbolPromotion();

    /**
     * @brief Returns the name of the pass
     * @return The string "SymbolPromotion"
     */
    std::string name() override;

    /**
     * @brief Executes the symbol promotion pass
     *
     * Traverses the structured SDFG and applies symbol promotion to all
     * eligible blocks containing single-tasklet computations.
     *
     * @param builder The SDFG builder
     * @param analysis_manager The analysis manager
     * @return true if any transformations were applied, false otherwise
     */
    virtual bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    /**
     * @brief Converts a tasklet input operand to a symbolic expression
     *
     * Creates a symbolic expression for a tasklet input by examining its source:
     * - For constant nodes, returns an integer literal
     * - For access nodes, returns a symbolic variable
     *
     * @param dataflow The dataflow graph containing the tasklet
     * @param tasklet The tasklet whose input to convert
     * @param op The name of the input connector
     * @return A symbolic expression representing the input, or null if not found
     */
    static symbolic::Expression
    as_symbol(const data_flow::DataFlowGraph& dataflow, const data_flow::Tasklet& tasklet, const std::string& op);
};

} // namespace passes
} // namespace sdfg
