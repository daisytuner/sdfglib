#pragma once

#include <cstddef>

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/data_flow/data_flow_node.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/element.h"
#include "sdfg/function.h"
#include "sdfg/graph/graph.h"
#include "sdfg/memory//offloading_node.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace cuda {

inline data_flow::LibraryNodeCode LibraryNodeType_CUDA_Offloading("CUDAOffloading");

class CUDAOffloadingNode : public memory::OffloadingNode {
private:
    symbolic::Expression device_id_;

public:
    CUDAOffloadingNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        symbolic::Expression size,
        symbolic::Expression device_id,
        memory::DataTransferDirection transfer_direction,
        memory::BufferLifecycle buffer_lifecycle
    );

    void validate(const Function& function) const override;

    const symbolic::Expression device_id() const;

    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;

    symbolic::SymbolSet symbols() const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    virtual bool blocking() const override;

    virtual bool redundant_with(const memory::OffloadingNode& other) const override;

    virtual bool equal_with(const memory::OffloadingNode& other) const override;
};

class CUDAOffloadingNodeDispatcher : public codegen::LibraryNodeDispatcher {
public:
    CUDAOffloadingNodeDispatcher(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const data_flow::LibraryNode& node
    );

    void dispatch_code(
        codegen::PrettyPrinter& stream,
        codegen::PrettyPrinter& globals_stream,
        codegen::CodeSnippetFactory& library_snippet_factory
    ) override;

    virtual codegen::InstrumentationInfo instrumentation_info() const override;
};

class CUDAOffloadingNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const sdfg::data_flow::LibraryNode& library_node) override;

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j,
        sdfg::builder::StructuredSDFGBuilder& builder,
        sdfg::structured_control_flow::Block& parent
    ) override;
};

} // namespace cuda
} // namespace sdfg
