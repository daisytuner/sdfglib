/**
 * @file expansion_pass.h
 * @brief Library node expansion pass
 *
 * This file defines the expansion pass that transforms library nodes with
 * ImplementationType_NONE into primitive SDFG operations. The expansion pass
 * visits all blocks in the SDFG and attempts to expand math library nodes.
 *
 * ## Expansion Process
 *
 * The expansion pass:
 * 1. Iterates through all library nodes in each block
 * 2. Skips nodes with a specific implementation type (not NONE)
 * 3. For math nodes with ImplementationType_NONE, calls their expand() method
 * 4. The expand() method transforms the high-level operation into maps, tasklets, etc.
 *
 * The pass is applied iteratively until no more expansions occur, allowing
 * multi-level expansions where one library node expands into others.
 *
 * @see math::MathNode::expand for node-specific expansion logic
 * @see passes::Pass for the pass interface
 */

#pragma once

#include "sdfg/passes/pass.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

/**
 * @class Expansion
 * @brief Visitor that expands library nodes into primitive operations
 *
 * The Expansion visitor traverses the SDFG and expands library nodes that
 * have ImplementationType_NONE. This allows high-level mathematical operations
 * to be transformed into lower-level constructs that can be optimized and
 * scheduled.
 */
class Expansion : public visitor::StructuredSDFGVisitor {
public:
    /**
     * @brief Construct the expansion visitor
     * @param builder SDFG builder for creating new nodes
     * @param analysis_manager Analysis manager for querying properties
     */
    Expansion(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    /**
     * @brief Get the pass name
     * @return Name of the pass
     */
    static std::string name() { return "Expansion"; };

    /**
     * @brief Visit a block and attempt to expand its library nodes
     * @param node Block to visit
     * @return True if any expansion occurred
     */
    bool accept(structured_control_flow::Block& node) override;
};

/**
 * @typedef ExpansionPass
 * @brief Pass wrapper for the Expansion visitor
 *
 * This typedef creates a pass from the Expansion visitor, allowing it to be
 * used in the pass pipeline system.
 */
typedef VisitorPass<Expansion> ExpansionPass;

} // namespace passes
} // namespace sdfg
