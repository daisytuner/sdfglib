/**
 * @file block_dispatcher.h
 * @brief Dispatchers for dataflow blocks and library nodes
 *
 * This file defines dispatchers for generating code from dataflow blocks and
 * library nodes. It includes:
 * - BlockDispatcher: Generates code for blocks containing dataflow
 * - DataFlowDispatcher: Generates code for dataflow graphs
 * - LibraryNodeDispatcher: Base class for library node code generation
 *
 * ## Library Node Dispatchers
 *
 * LibraryNodeDispatcher is the base class for generating code from library nodes.
 * Each library node type can have a custom dispatcher registered in the
 * LibraryNodeDispatcherRegistry. The dispatcher is responsible for:
 * - Generating the appropriate library call or inline code
 * - Handling input/output data access
 * - Managing implementation-specific details
 *
 * Dispatchers work together with the library node's implementation_type:
 * - If implementation_type is NONE, the node may be expanded first
 * - If implementation_type specifies a library (e.g., BLAS), the dispatcher
 *   generates a call to that library
 *
 * @see node_dispatcher_registry.h for dispatcher registration
 * @see data_flow::LibraryNode for library node definition
 */

#pragma once

#include "sdfg/codegen/dispatchers/node_dispatcher.h"
#include "sdfg/data_flow/library_node.h"

namespace sdfg {
namespace codegen {

class BlockDispatcher : public NodeDispatcher {
private:
    const structured_control_flow::Block& node_;

public:
    BlockDispatcher(
        LanguageExtension& language_extension,
        StructuredSDFG& sdfg,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::Block& node,
        InstrumentationPlan& instrumentation_plan,
        ArgCapturePlan& arg_capture_plan
    );

    void dispatch_node(
        PrettyPrinter& main_stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory
    ) override;
};

class DataFlowDispatcher {
private:
    LanguageExtension& language_extension_;
    const Function& function_;
    const data_flow::DataFlowGraph& data_flow_graph_;
    const InstrumentationPlan& instrumentation_plan_;

    void dispatch_ref(PrettyPrinter& stream, const data_flow::Memlet& memlet);

    void dispatch_deref_src(PrettyPrinter& stream, const data_flow::Memlet& memlet);

    void dispatch_deref_dst(PrettyPrinter& stream, const data_flow::Memlet& memlet);

    void dispatch_tasklet(PrettyPrinter& stream, const data_flow::Tasklet& tasklet);

    void dispatch_library_node(
        PrettyPrinter& stream,
        PrettyPrinter& globals_stream,
        CodeSnippetFactory& library_snippet_factory,
        const data_flow::LibraryNode& libnode
    );

public:
    DataFlowDispatcher(
        LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const InstrumentationPlan& instrumentation_plan
    );

    void dispatch(PrettyPrinter& stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory);
};

/**
 * @class LibraryNodeDispatcher
 * @brief Base class for library node code generation dispatchers
 *
 * LibraryNodeDispatcher provides the interface for generating code from library
 * nodes. Subclasses implement specific code generation for different library
 * operations (BLAS calls, memory operations, etc.).
 *
 * The dispatcher has access to:
 * - The library node being dispatched
 * - The containing dataflow graph
 * - The function context
 * - The language extension for code generation
 *
 * Subclasses override dispatch_code() to generate the actual operation code.
 */
class LibraryNodeDispatcher {
protected:
    LanguageExtension& language_extension_; ///< Language extension for code generation
    const Function& function_; ///< Function context
    const data_flow::DataFlowGraph& data_flow_graph_; ///< Containing dataflow graph
    const data_flow::LibraryNode& node_; ///< Library node being dispatched

public:
    /**
     * @brief Construct a library node dispatcher
     * @param language_extension Language extension for code generation
     * @param function Function context
     * @param data_flow_graph Containing dataflow graph
     * @param node Library node to dispatch
     */
    LibraryNodeDispatcher(
        LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const data_flow::LibraryNode& node
    );

    virtual ~LibraryNodeDispatcher() = default;

    /**
     * @brief Begin code generation for the node
     * @param stream Output stream for generated code
     * @return True if a declaration was generated
     */
    virtual bool begin_node(PrettyPrinter& stream) { return false; }

    /**
     * @brief End code generation for the node
     * @param stream Output stream for generated code
     * @param has_declaration Whether a declaration was generated
     */
    virtual void end_node(PrettyPrinter& stream, bool has_declaration) {}

    /**
     * @brief Dispatch the library node to code
     * @param stream Main code stream
     * @param globals_stream Global declarations stream
     * @param library_snippet_factory Factory for library code snippets
     */
    virtual void
    dispatch(PrettyPrinter& stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory);

    /**
     * @brief Generate the operation-specific code
     *
     * Subclasses override this method to generate the actual library call or
     * inline implementation. This might include:
     * - BLAS library calls (cblas_dgemm, etc.)
     * - Standard library calls (memcpy, malloc, etc.)
     * - Custom inline implementations
     *
     * @param stream Main code stream
     * @param globals_stream Global declarations stream
     * @param library_snippet_factory Factory for library code snippets
     */
    virtual void
    dispatch_code(PrettyPrinter& stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory) {}

    /**
     * @brief Get instrumentation information for this node
     * @return Instrumentation information
     */
    virtual InstrumentationInfo instrumentation_info() const;
};

} // namespace codegen
} // namespace sdfg
