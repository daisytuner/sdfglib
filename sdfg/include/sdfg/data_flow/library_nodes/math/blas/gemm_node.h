#pragma once

#include "sdfg/data_flow/library_nodes/math/math_node.h"

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/serializer/json_serializer.h"

#include "sdfg/data_flow/library_nodes/math/math.h"

namespace sdfg {
namespace math {
namespace blas {

inline data_flow::LibraryNodeCode LibraryNodeType_GEMM("GEMM");

class GEMMNode : public BLASNode {
private:
    BLAS_Layout layout_;
    BLAS_Transpose trans_a_;
    BLAS_Transpose trans_b_;

    symbolic::Expression m_;
    symbolic::Expression n_;
    symbolic::Expression k_;
    symbolic::Expression lda_;
    symbolic::Expression ldb_;
    symbolic::Expression ldc_;

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
        symbolic::Expression ldc
    );

    BLAS_Layout layout() const;

    BLAS_Transpose trans_a() const;

    BLAS_Transpose trans_b() const;

    symbolic::Expression m() const;

    symbolic::Expression n() const;

    symbolic::Expression k() const;

    symbolic::Expression lda() const;

    symbolic::Expression ldb() const;

    symbolic::Expression ldc() const;

    void validate(const Function& function) const override;

    bool expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    symbolic::SymbolSet symbols() const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;

    std::string toStr() const override;

    symbolic::Expression flop() const override;

    symbolic::Expression flops(
        symbolic::Condition alpha_non_zero,
        symbolic::Condition alpha_non_ident,
        symbolic::Condition beta_non_zero,
        symbolic::Condition beta_non_ident
    ) const;
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

    void dispatch_code(
        codegen::PrettyPrinter& stream,
        codegen::PrettyPrinter& globals_stream,
        codegen::CodeSnippetFactory& library_snippet_factory
    ) override;
};

} // namespace blas
} // namespace math
} // namespace sdfg
