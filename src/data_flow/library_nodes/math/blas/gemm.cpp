#include "sdfg/data_flow/library_nodes/math/blas/gemm.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

#include "sdfg/analysis/scope_analysis.h"

namespace sdfg {
namespace math {
namespace blas {

GEMMNode::GEMMNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
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
    const std::string& alpha,
    const std::string& beta
)
    : MathNode(element_id, debug_info, vertex, parent, LibraryNodeType_GEMM, {"C"}, {"A", "B", "C"}),
      precision_(precision), layout_(layout), trans_a_(trans_a), trans_b_(trans_b), m_(m), n_(n), k_(k), lda_(lda),
      ldb_(ldb), ldc_(ldc), alpha_(alpha), beta_(beta) {
    if (alpha.empty()) {
        this->inputs_.push_back("alpha");
    }
    if (beta.empty()) {
        this->inputs_.push_back("beta");
    }
}

BLAS_Precision GEMMNode::precision() const { return this->precision_; };

BLAS_Layout GEMMNode::layout() const { return this->layout_; };

BLAS_Transpose GEMMNode::trans_a() const { return this->trans_a_; };

BLAS_Transpose GEMMNode::trans_b() const { return this->trans_b_; };

symbolic::Expression GEMMNode::m() const { return this->m_; };

symbolic::Expression GEMMNode::n() const { return this->n_; };

symbolic::Expression GEMMNode::k() const { return this->k_; };

symbolic::Expression GEMMNode::lda() const { return this->lda_; };

symbolic::Expression GEMMNode::ldb() const { return this->ldb_; };

symbolic::Expression GEMMNode::ldc() const { return this->ldc_; };

std::string GEMMNode::alpha() const { return this->alpha_; };

std::string GEMMNode::beta() const { return this->beta_; };

void GEMMNode::validate(const Function& function) const {
    auto& graph = this->get_parent();

    if (graph.in_degree(*this) != this->inputs_.size()) {
        throw InvalidSDFGException("GEMMNode must have " + std::to_string(this->inputs_.size()) + " inputs");
    }
    if (graph.out_degree(*this) != 1) {
        throw InvalidSDFGException("GEMMNode must have 1 output");
    }

    // Check if all inputs are connected A, B, C, (alpha), (beta)
    std::unordered_map<std::string, const data_flow::Memlet*> memlets;
    for (auto& input : this->inputs_) {
        bool found = false;
        for (auto& iedge : graph.in_edges(*this)) {
            if (iedge.dst_conn() == input) {
                found = true;
                memlets[input] = &iedge;
                break;
            }
        }
        if (!found) {
            throw InvalidSDFGException("GEMMNode input " + input + " not found");
        }
    }

    // Check if output is connected to C
    auto& oedge = *graph.out_edges(*this).begin();
    if (oedge.src_conn() != this->outputs_.at(0)) {
        throw InvalidSDFGException("GEMMNode output " + this->outputs_.at(0) + " not found");
    }

    // Check dimensions of A, B, C
    auto& a_memlet = memlets.at("A");
    auto& a_subset_begin = a_memlet->begin_subset();
    auto& a_subset_end = a_memlet->end_subset();
    if (a_subset_begin.size() != 1) {
        throw InvalidSDFGException("GEMMNode input A must have 1 dimensions");
    }
    data_flow::Subset a_dims;
    for (size_t i = 0; i < a_subset_begin.size(); i++) {
        a_dims.push_back(symbolic::sub(a_subset_end[i], a_subset_begin[i]));
    }

    auto& b_memlet = memlets.at("B");
    auto& b_subset_begin = b_memlet->begin_subset();
    auto& b_subset_end = b_memlet->end_subset();
    if (b_subset_begin.size() != 1) {
        throw InvalidSDFGException("GEMMNode input B must have 1 dimensions");
    }
    data_flow::Subset b_dims;
    for (size_t i = 0; i < b_subset_begin.size(); i++) {
        b_dims.push_back(symbolic::sub(b_subset_end[i], b_subset_begin[i]));
    }

    auto& c_memlet = memlets.at("C");
    auto& c_subset_begin = c_memlet->begin_subset();
    auto& c_subset_end = c_memlet->end_subset();
    if (c_subset_begin.size() != 1) {
        throw InvalidSDFGException("GEMMNode input C must have 1 dimensions");
    }
    data_flow::Subset c_dims;
    for (size_t i = 0; i < c_subset_begin.size(); i++) {
        c_dims.push_back(symbolic::sub(c_subset_end[i], c_subset_begin[i]));
    }

    // TODO: Check if dimensions of A, B, C are valid
}

bool GEMMNode::expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = builder.subject();
    auto& dataflow = this->get_parent();
    auto& block = static_cast<structured_control_flow::Block&>(*dataflow.get_parent());

    // TODO: Expand GEMM node

    return false;
}

std::unique_ptr<data_flow::DataFlowNode> GEMMNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    auto node_clone = std::unique_ptr<GEMMNode>(new GEMMNode(
        element_id,
        this->debug_info(),
        vertex,
        parent,
        this->precision_,
        this->layout_,
        this->trans_a_,
        this->trans_b_,
        this->m_,
        this->n_,
        this->k_,
        this->lda_,
        this->ldb_,
        this->ldc_,
        this->alpha_,
        this->beta_
    ));
    node_clone->implementation_type() = this->implementation_type();
    return std::move(node_clone);
}

nlohmann::json GEMMNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const GEMMNode& gemm_node = static_cast<const GEMMNode&>(library_node);
    nlohmann::json j;

    serializer::JSONSerializer serializer;
    j["code"] = gemm_node.code().value();
    j["precision"] = gemm_node.precision();
    j["layout"] = gemm_node.layout();
    j["trans_a"] = gemm_node.trans_a();
    j["trans_b"] = gemm_node.trans_b();
    j["m"] = serializer.expression(gemm_node.m());
    j["n"] = serializer.expression(gemm_node.n());
    j["k"] = serializer.expression(gemm_node.k());
    j["lda"] = serializer.expression(gemm_node.lda());
    j["ldb"] = serializer.expression(gemm_node.ldb());
    j["ldc"] = serializer.expression(gemm_node.ldc());
    j["alpha"] = gemm_node.alpha();
    j["beta"] = gemm_node.beta();

    return j;
}

data_flow::LibraryNode& GEMMNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    // Assertions for required fields
    assert(j.contains("element_id"));
    assert(j.contains("code"));
    assert(j.contains("debug_info"));

    auto code = j["code"].get<std::string>();
    if (code != LibraryNodeType_GEMM.value()) {
        throw std::runtime_error("Invalid library node code");
    }

    // Extract debug info using JSONSerializer
    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    auto precision = j.at("precision").get<BLAS_Precision>();
    auto layout = j.at("layout").get<BLAS_Layout>();
    auto trans_a = j.at("trans_a").get<BLAS_Transpose>();
    auto trans_b = j.at("trans_b").get<BLAS_Transpose>();
    auto m = SymEngine::Expression(j.at("m"));
    auto n = SymEngine::Expression(j.at("n"));
    auto k = SymEngine::Expression(j.at("k"));
    auto lda = SymEngine::Expression(j.at("lda"));
    auto ldb = SymEngine::Expression(j.at("ldb"));
    auto ldc = SymEngine::Expression(j.at("ldc"));
    auto alpha = j.at("alpha").get<std::string>();
    auto beta = j.at("beta").get<std::string>();

    return builder.add_library_node<
        GEMMNode>(parent, debug_info, precision, layout, trans_a, trans_b, m, n, k, lda, ldb, ldc, alpha, beta);
}

GEMMNodeDispatcher_BLAS::GEMMNodeDispatcher_BLAS(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const GEMMNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void GEMMNodeDispatcher_BLAS::dispatch(codegen::PrettyPrinter& stream) {
    stream << "// IMPLEMENT DGEMM HERE" << std::endl;
}

GEMMNodeDispatcher_CUBLAS::GEMMNodeDispatcher_CUBLAS(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const GEMMNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void GEMMNodeDispatcher_CUBLAS::dispatch(codegen::PrettyPrinter& stream) {
    throw std::runtime_error("GEMMNodeDispatcher_CUBLAS not implemented");
}

} // namespace blas
} // namespace math
} // namespace sdfg
