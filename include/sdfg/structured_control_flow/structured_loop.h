#pragma once

#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {

namespace builder {
class StructuredSDFGBuilder;
}

namespace structured_control_flow {

/**
 * @brief Base class for structured loop constructs
 *
 * StructuredLoop is the abstract base class for all structured loop types in
 * a StructuredSDFG. It provides common functionality for loops with:
 * - An induction variable (loop counter)
 * - An initialization expression
 * - An update expression (how the induction variable changes each iteration)
 * - A loop condition (when to continue iterating)
 * - A body sequence containing the loop's control flow
 *
 * Derived loop types include:
 * - For: Traditional for-loops with explicit initialization, condition, and update
 * - Map: Parallel loops that can be mapped to parallel execution
 *
 * **Loop Structure:**
 * ```
 * indvar = init
 * while (condition):
 *   <body sequence>
 *   indvar = update
 * ```
 *
 * The loop body is a Sequence that can contain any control flow nodes (blocks,
 * nested loops, conditionals, etc.).
 *
 * @see For
 * @see Map
 * @see Sequence
 */
class StructuredLoop : public ControlFlowNode {
    friend class sdfg::builder::StructuredSDFGBuilder;

protected:
    symbolic::Symbol indvar_;
    symbolic::Expression init_;
    symbolic::Expression update_;
    symbolic::Condition condition_;

    std::unique_ptr<Sequence> root_;

    StructuredLoop(
        size_t element_id,
        const DebugInfo& debug_info,
        symbolic::Symbol indvar,
        symbolic::Expression init,
        symbolic::Expression update,
        symbolic::Condition condition
    );

public:
    virtual ~StructuredLoop() = default;

    StructuredLoop(const StructuredLoop& node) = delete;
    StructuredLoop& operator=(const StructuredLoop&) = delete;

    void validate(const Function& function) const override;

    /**
     * @brief Get the induction variable (loop counter) symbol
     * @return The loop's induction variable
     */
    const symbolic::Symbol indvar() const;

    /**
     * @brief Get the initialization expression for the induction variable
     * @return Expression evaluated to initialize the induction variable
     */
    const symbolic::Expression init() const;

    /**
     * @brief Get the update expression for the induction variable
     * @return Expression evaluated to update the induction variable each iteration
     */
    const symbolic::Expression update() const;

    /**
     * @brief Get the loop continuation condition
     * @return Boolean expression evaluated before each iteration; loop continues while true
     */
    const symbolic::Condition condition() const;

    /**
     * @brief Access the loop body sequence
     * @return Reference to the sequence containing the loop body
     */
    Sequence& root() const;

    /**
     * @brief Replace occurrences of an expression in loop parameters and body
     * @param old_expression Expression to replace
     * @param new_expression Expression to replace with
     */
    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;
};

} // namespace structured_control_flow
} // namespace sdfg
