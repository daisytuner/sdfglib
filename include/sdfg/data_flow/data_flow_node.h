/**
 * @file data_flow_node.h
 * @brief Base class for all nodes in the dataflow graph
 *
 * DataFlowNode is the abstract base class for all nodes that can appear in a dataflow
 * graph. It provides the fundamental interface for nodes including vertex management,
 * parent graph access, validation, and cloning.
 *
 * ## Key Concepts
 *
 * ### Node Hierarchy
 * The node hierarchy is:
 * - DataFlowNode (abstract base)
 *   - AccessNode: Data access points (variables, arrays)
 *     - ConstantNode: Constant literal values
 *   - CodeNode: Computational operations (abstract)
 *     - Tasklet: Simple operations (add, mul, etc.)
 *     - LibraryNode: Complex operations (BLAS, etc.)
 *
 * ### Graph Vertex
 * Each node is associated with a Boost graph vertex that represents its position
 * in the dataflow graph. The vertex is used for graph traversal and edge management.
 *
 * ### Parent Graph
 * Each node maintains a reference to its parent DataFlowGraph, which owns the node
 * and manages its lifetime and connections.
 *
 * ## Example Usage
 *
 * Working with nodes through the base interface:
 * @code
 * // Get the node's vertex for graph operations
 * graph::Vertex v = node.vertex();
 *
 * // Get the parent dataflow graph
 * DataFlowGraph& graph = node.get_parent();
 *
 * // Check incoming/outgoing edges
 * size_t in_degree = graph.in_degree(node);
 * size_t out_degree = graph.out_degree(node);
 *
 * // Validate the node
 * node.validate(function);
 * @endcode
 *
 * @see AccessNode for data access nodes
 * @see CodeNode for computational nodes
 * @see DataFlowGraph for the container graph
 * @see Element for the base element class
 */

#pragma once

#include <boost/lexical_cast.hpp>
#include <nlohmann/json.hpp>

#include "sdfg/element.h"
#include "sdfg/graph/graph.h"

using json = nlohmann::json;

namespace sdfg {

namespace builder {
class SDFGBuilder;
class StructuredSDFGBuilder;
} // namespace builder

namespace data_flow {

class DataFlowGraph;

/**
 * @class DataFlowNode
 * @brief Abstract base class for all dataflow graph nodes
 *
 * DataFlowNode provides the core interface for all nodes in the dataflow graph.
 * Key responsibilities:
 * - Vertex management: Each node has a unique graph vertex
 * - Parent graph access: Nodes know their containing graph
 * - Validation: Abstract interface for semantic validation
 * - Cloning: Abstract interface for creating node copies
 *
 * This is an abstract class and cannot be instantiated directly.
 * Use derived classes like AccessNode, Tasklet, or LibraryNode.
 */
class DataFlowNode : public Element {
    friend class sdfg::builder::SDFGBuilder;
    friend class sdfg::builder::StructuredSDFGBuilder;

private:
    // Remark: Exclusive resource
    graph::Vertex vertex_; ///< Graph vertex representing this node's position

    DataFlowGraph* parent_; ///< Parent dataflow graph that owns this node

protected:
    /**
     * @brief Protected constructor for dataflow nodes
     * @param element_id Unique element identifier
     * @param debug_info Debug information for this node
     * @param vertex Graph vertex for this node
     * @param parent Parent dataflow graph
     */
    DataFlowNode(size_t element_id, const DebugInfo& debug_info, const graph::Vertex vertex, DataFlowGraph& parent);

public:
    // Remark: Exclusive resource
    DataFlowNode(const DataFlowNode& data_node) = delete;
    DataFlowNode& operator=(const DataFlowNode&) = delete;

    /**
     * @brief Get the graph vertex for this node
     * @return The Boost graph vertex representing this node
     */
    graph::Vertex vertex() const;

    /**
     * @brief Get the parent dataflow graph (const)
     * @return Const reference to the parent DataFlowGraph
     */
    const DataFlowGraph& get_parent() const;

    /**
     * @brief Get the parent dataflow graph (mutable)
     * @return Mutable reference to the parent DataFlowGraph
     */
    DataFlowGraph& get_parent();

    /**
     * @brief Clone this node for graph transformations
     * @param element_id New element identifier for the clone
     * @param vertex New graph vertex for the clone
     * @param parent Parent graph for the clone
     * @return Unique pointer to the cloned node
     *
     * Pure virtual function that must be implemented by derived classes
     * to support graph transformations and optimizations.
     */
    virtual std::unique_ptr<DataFlowNode> clone(size_t element_id, const graph::Vertex vertex, DataFlowGraph& parent)
        const = 0;
};
} // namespace data_flow
} // namespace sdfg
