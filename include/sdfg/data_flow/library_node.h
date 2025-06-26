#pragma once

#include <string>
#include <vector>

#include "sdfg/data_flow/code_node.h"
#include "sdfg/graph/graph.h"

namespace sdfg {

namespace builder {
class SDFGBuilder;
class StructuredSDFGBuilder;
}  // namespace builder

namespace data_flow {

typedef StringEnum LibraryNodeCode;

class LibraryNode : public CodeNode {
    friend class sdfg::builder::SDFGBuilder;
    friend class sdfg::builder::StructuredSDFGBuilder;

   protected:
    LibraryNodeCode code_;
    std::vector<std::string> outputs_;
    std::vector<std::string> inputs_;
    bool side_effect_;

    LibraryNode(size_t element_id, const DebugInfo& debug_info, const graph::Vertex vertex,
                DataFlowGraph& parent, const LibraryNodeCode& code,
                const std::vector<std::string>& outputs, const std::vector<std::string>& inputs,
                const bool side_effect);

   public:
    LibraryNode(const LibraryNode& data_node) = delete;
    LibraryNode& operator=(const LibraryNode&) = delete;

    virtual ~LibraryNode() = default;

    const LibraryNodeCode& code() const;

    const std::vector<std::string>& inputs() const;

    const std::vector<std::string>& outputs() const;

    const std::string& input(size_t index) const;

    const std::string& output(size_t index) const;

    bool side_effect() const;

    bool needs_connector(size_t index) const override;

    virtual std::string toStr() const;

    virtual symbolic::SymbolSet symbols() const = 0;
};

/*
 * List of predefiened library node codes.
 *
 */

inline data_flow::LibraryNodeCode BARRIER_LOCAL{"barrier_local"};

}  // namespace data_flow
}  // namespace sdfg
