#pragma once

#include <string>
#include <vector>

#include "sdfg/data_flow/code_node.h"
#include "sdfg/debug_info.h"
#include "sdfg/exceptions.h"
#include "sdfg/graph/graph.h"

namespace sdfg {

namespace builder {
class SDFGBuilder;
class StructuredSDFGBuilder;
} // namespace builder

namespace data_flow {

typedef StringEnum LibraryNodeCode;
typedef StringEnum ImplementationType;
inline ImplementationType ImplementationType_NONE{""};

class LibraryNode : public CodeNode {
    friend class sdfg::builder::SDFGBuilder;
    friend class sdfg::builder::StructuredSDFGBuilder;

protected:
    LibraryNodeCode code_;
    bool side_effect_;

    ImplementationType implementation_type_;

    LibraryNode(
        size_t element_id,
        const DebugInfoRegion& debug_info_region,
        const graph::Vertex vertex,
        DataFlowGraph& parent,
        const LibraryNodeCode& code,
        const std::vector<std::string>& outputs,
        const std::vector<std::string>& inputs,
        const bool side_effect,
        const ImplementationType& implementation_type
    );

public:
    LibraryNode(const LibraryNode& data_node) = delete;
    LibraryNode& operator=(const LibraryNode&) = delete;

    virtual ~LibraryNode() = default;

    const LibraryNodeCode& code() const;

    const ImplementationType& implementation_type() const;

    ImplementationType& implementation_type();

    bool side_effect() const;

    bool needs_connector(size_t index) const override;

    virtual std::string toStr() const;

    virtual symbolic::SymbolSet symbols() const = 0;
};

} // namespace data_flow
} // namespace sdfg
