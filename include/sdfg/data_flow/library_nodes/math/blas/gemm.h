#pragma once

#include "sdfg/data_flow/library_nodes/math/math_node.h"

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/serializer/json_serializer.h"

#include "sdfg/data_flow/library_nodes/math/blas/blas.h"

namespace sdfg {
namespace math {
namespace blas {

inline data_flow::LibraryNodeCode LibraryNodeType_GEMM("GEMM");

class GEMMNode : public math::MathNode {
private:
    BLAS_Precision precision_;
    BLAS_Layout layout_;
    BLAS_Transpose trans_a_;
    BLAS_Transpose trans_b_;

    symbolic::Expression m_;
    symbolic::Expression n_;
    symbolic::Expression k_;
    symbolic::Expression lda_;
    symbolic::Expression ldb_;
    symbolic::Expression ldc_;

    std::string alpha_;
    std::string beta_;

public:
    GEMMNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const data_flow::ImplementationType& implementation_type,
        const BLAS_Precision& precision,
        const BLAS_Layout& layout,
        const BLAS_Transpose& trans_a,
        const BLAS_Transpose& trans_b,
        symbolic::Expression m,
        symbolic::Expression n,
        symbolic::Expression k,
        symbolic::Expression lda,
        symbolic::Expression ldb,
        symbolic::Expression ldc,
        const std::string& alpha = "",
        const std::string& beta = ""
    );

    BLAS_Precision precision() const;

    BLAS_Layout layout() const;

    BLAS_Transpose trans_a() const;

    BLAS_Transpose trans_b() const;

    symbolic::Expression m() const;

    symbolic::Expression n() const;

    symbolic::Expression k() const;

    symbolic::Expression lda() const;

    symbolic::Expression ldb() const;

    symbolic::Expression ldc() const;

    std::string alpha() const;

    std::string beta() const;

    void validate(const Function& function) const override;

    bool expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;
};

class GEMMNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

class GEMMNodeDispatcher_BLAS : public codegen::LibraryNodeDispatcher {
public:
    GEMMNodeDispatcher_BLAS(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const GEMMNode& node
    );

    void dispatch(codegen::PrettyPrinter& stream) override;
};

class GEMMNodeDispatcher_CUBLAS : public codegen::LibraryNodeDispatcher {
public:
    GEMMNodeDispatcher_CUBLAS(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const GEMMNode& node
    );

    void dispatch(codegen::PrettyPrinter& stream) override;
};

} // namespace blas
} // namespace math
} // namespace sdfg
