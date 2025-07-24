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
    const std::string& alpha,
    const std::string& beta
)
    : MathNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_GEMM,
          {"C"},
          {"A", "B", "C", alpha, beta},
          implementation_type
      ),
      precision_(precision), layout_(layout), trans_a_(trans_a), trans_b_(trans_b), m_(m), n_(n), k_(k), lda_(lda),
      ldb_(ldb), ldc_(ldc) {}

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

const std::string& GEMMNode::alpha() const { return this->inputs_.at(3); };

const std::string& GEMMNode::beta() const { return this->inputs_.at(4); };

void GEMMNode::validate(const Function& function) const {
    auto& graph = this->get_parent();

    if (this->inputs_.size() != 5) {
        throw InvalidSDFGException("GEMMNode must have 5 inputs: A, B, C, (alpha), (beta)");
    }

    int input_edge_count = graph.in_degree(*this);
    if (input_edge_count < 3 || input_edge_count > 5) {
        throw InvalidSDFGException("GEMMNode must have 3-5 inputs");
    }
    if (graph.out_degree(*this) != 1) {
        throw InvalidSDFGException("GEMMNode must have 1 output");
    }

    // // Check if all inputs are connected A, B, C, (alpha), (beta)
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
        if (!found && (input == "A" || input == "B" || input == "C")) {
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

types::PrimitiveType GEMMNode::scalar_primitive() const {
    switch (this->precision_) {
        case BLAS_Precision::s:
            return types::PrimitiveType::Float;
        case BLAS_Precision::d:
            return types::PrimitiveType::Double;
        case BLAS_Precision::h:
            return types::PrimitiveType::Half;
        default:
            return types::PrimitiveType::Void;
    }
}

bool GEMMNode::expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = builder.subject();
    auto& dataflow = this->get_parent();
    auto& block = static_cast<structured_control_flow::Block&>(*dataflow.get_parent());

    auto& scope_analyisis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto& parent = static_cast<structured_control_flow::Sequence&>(*scope_analyisis.parent_scope(&block));

    if (trans_a_ != BLAS_Transpose::No || trans_b_ != BLAS_Transpose::No) {
        return false;
    }

    auto& alpha = this->alpha();
    auto& beta = this->beta();

    auto primitive_type = scalar_primitive();
    if (primitive_type == types::PrimitiveType::Void) {
        return false;
    }

    types::Scalar scalar_type(primitive_type);

    auto in_edges = dataflow.in_edges(*this);
    auto in_edges_it = in_edges.begin();

    data_flow::Memlet* iedge_a = nullptr;
    data_flow::Memlet* iedge_b = nullptr;
    data_flow::Memlet* iedge_c = nullptr;
    data_flow::Memlet* alpha_edge = nullptr;
    data_flow::Memlet* beta_edge = nullptr;
    while (in_edges_it != in_edges.end()) {
        auto& edge = *in_edges_it;
        auto dst_conn = edge.dst_conn();
        if (dst_conn == "A") {
            iedge_a = &edge;
        } else if (dst_conn == "B") {
            iedge_b = &edge;
        } else if (dst_conn == "C") {
            iedge_c = &edge;
        } else if (dst_conn == alpha) {
            alpha_edge = &edge;
        } else if (dst_conn == beta) {
            beta_edge = &edge;
        } else {
            throw InvalidSDFGException("GEMMNode has unexpected input: " + dst_conn);
        }
        ++in_edges_it;
    }

    auto& oedge = *dataflow.out_edges(*this).begin();

    // Checks if legal
    auto* input_node_a = dynamic_cast<data_flow::AccessNode*>(&iedge_a->src());
    auto* input_node_b = dynamic_cast<data_flow::AccessNode*>(&iedge_b->src());
    auto* input_node_c = dynamic_cast<data_flow::AccessNode*>(&iedge_c->src());
    auto* output_node = dynamic_cast<data_flow::AccessNode*>(&oedge.dst());
    data_flow::AccessNode* alpha_node = nullptr;
    data_flow::AccessNode* beta_node = nullptr;

    if (alpha_edge) {
        alpha_node = dynamic_cast<data_flow::AccessNode*>(&alpha_edge->src());
    }
    if (beta_edge) {
        beta_node = dynamic_cast<data_flow::AccessNode*>(&beta_edge->src());
    }

    // we must be the only thing in this block, as we do not support splitting a block into pre, expanded lib-node, post
    if (!input_node_a || dataflow.in_degree(*input_node_a) != 0 || !input_node_b ||
        dataflow.in_degree(*input_node_b) != 0 || !input_node_c || dataflow.in_degree(*input_node_c) != 0 ||
        !output_node || dataflow.out_degree(*output_node) != 0) {
        return false; // data nodes are not standalone
    }
    if ((alpha_node && dataflow.in_degree(*alpha_node) != 0) || (beta_node && dataflow.in_degree(*beta_node) != 0)) {
        return false; // alpha and beta are not standalone
    }
    for (auto* nd : dataflow.data_nodes()) {
        if (nd != input_node_a && nd != input_node_b && nd != input_node_c && nd != output_node &&
            (!alpha_node || nd != alpha_node) && (!beta_node || nd != beta_node)) {
            return false; // there are other nodes in here that we could not preserve correctly
        }
    }

    auto& A_var = input_node_a->data();
    auto& B_var = input_node_b->data();
    auto& C_in_var = input_node_c->data();
    auto& C_out_var = output_node->data();


    // Add new graph after the current block
    auto& new_sequence = builder.add_sequence_before(parent, block, block.debug_info()).first;

    // Add maps
    std::vector<symbolic::Expression> indvar_ends{this->m(), this->n(), this->k()};
    auto& begin_subsets_out = oedge.begin_subset();
    auto& end_subsets_out = oedge.end_subset();
    auto& begin_subsets_in_a = iedge_a->begin_subset();
    auto& end_subsets_in_a = iedge_a->end_subset();
    data_flow::Subset new_subset;
    structured_control_flow::Sequence* last_scope = &new_sequence;
    structured_control_flow::Map* last_map = nullptr;
    structured_control_flow::Map* output_loop = nullptr;
    std::vector<std::string> indvar_names{"_i", "_j", "_k"};

    std::string sum_var = builder.find_new_name("_sum");
    builder.add_container(sum_var, scalar_type);

    for (size_t i = 0; i < 3; i++) {
        auto dim_begin = symbolic::zero();
        auto& dim_end = indvar_ends[i];

        std::string indvar_str = builder.find_new_name(indvar_names[i]);
        builder.add_container(indvar_str, types::Scalar(types::PrimitiveType::UInt64));

        auto indvar = symbolic::symbol(indvar_str);
        auto init = dim_begin;
        auto update = symbolic::add(indvar, symbolic::one());
        auto condition = symbolic::Lt(indvar, dim_end);
        last_map = &builder.add_map(
            *last_scope,
            indvar,
            condition,
            init,
            update,
            structured_control_flow::ScheduleType_Sequential,
            {},
            block.debug_info()
        );
        last_scope = &last_map->root();

        if (i == 1) {
            output_loop = last_map;
        }

        new_subset.push_back(indvar);
    }


    // Add code
    auto& init_block = builder.add_block_before(output_loop->root(), *last_map, block.debug_info()).first;
    auto& sum_init = builder.add_access(init_block, sum_var, block.debug_info());

    auto& init_tasklet =
        builder
            .add_tasklet(init_block, data_flow::assign, {"_out", scalar_type}, {{"0.0", scalar_type}}, block.debug_info());

    builder.add_computational_memlet(init_block, init_tasklet, "_out", sum_init, {}, block.debug_info());

    auto& code_block = builder.add_block(*last_scope, {}, block.debug_info());
    auto& input_node_a_new = builder.add_access(code_block, A_var, input_node_a->debug_info());
    auto& input_node_b_new = builder.add_access(code_block, B_var, input_node_b->debug_info());

    auto& core_fma = builder.add_tasklet(
        code_block,
        data_flow::fma,
        {"_out", scalar_type},
        {{"_in1", scalar_type}, {"_in2", scalar_type}, {"_in3", scalar_type}},
        block.debug_info()
    );
    auto& sum_in = builder.add_access(code_block, sum_var, block.debug_info());
    auto& sum_out = builder.add_access(code_block, sum_var, block.debug_info());
    builder.add_computational_memlet(code_block, sum_in, core_fma, "_in3", {}, block.debug_info());

    symbolic::Expression a_idx = symbolic::add(symbolic::mul(lda(), new_subset[0]), new_subset[2]);
    builder.add_computational_memlet(code_block, input_node_a_new, core_fma, "_in1", {a_idx}, block.debug_info());
    symbolic::Expression b_idx = symbolic::add(symbolic::mul(ldb(), new_subset[2]), new_subset[1]);
    builder.add_computational_memlet(code_block, input_node_b_new, core_fma, "_in2", {b_idx}, block.debug_info());
    builder.add_computational_memlet(code_block, core_fma, "_out", sum_out, {}, block.debug_info());

    auto& flush_block = builder.add_block_after(output_loop->root(), *last_map, block.debug_info()).first;
    auto& sum_final = builder.add_access(flush_block, sum_var, block.debug_info());
    auto& input_node_c_new = builder.add_access(flush_block, C_in_var, input_node_c->debug_info());
    symbolic::Expression c_idx = symbolic::add(symbolic::mul(ldc(), new_subset[0]), new_subset[1]);

    auto& scale_sum_tasklet = builder.add_tasklet(
        flush_block,
        data_flow::mul,
        {"_out", scalar_type},
        {{"_in1", scalar_type}, {alpha, scalar_type}},
        block.debug_info()
    );
    builder.add_computational_memlet(flush_block, sum_final, scale_sum_tasklet, "_in1", {}, block.debug_info());
    if (alpha_node) {
        auto& alpha_node_new = builder.add_access(flush_block, alpha_node->data(), block.debug_info());
        builder.add_computational_memlet(flush_block, scale_sum_tasklet, alpha, alpha_node_new, {}, block.debug_info());
    }

    std::string scaled_sum_temp = builder.find_new_name("scaled_sum_temp");
    builder.add_container(scaled_sum_temp, scalar_type);
    auto& scaled_sum_final = builder.add_access(flush_block, scaled_sum_temp, block.debug_info());
    builder.add_computational_memlet(flush_block, scale_sum_tasklet, "_out", scaled_sum_final, {}, block.debug_info());

    auto& scale_input_tasklet = builder.add_tasklet(
        flush_block,
        data_flow::mul,
        {"_out", scalar_type},
        {{"_in1", scalar_type}, {beta, scalar_type}},
        block.debug_info()
    );
    builder
        .add_computational_memlet(flush_block, input_node_c_new, scale_input_tasklet, "_in1", {c_idx}, block.debug_info());
    if (beta_node) {
        auto& beta_node_new = builder.add_access(flush_block, beta_node->data(), block.debug_info());
        builder.add_computational_memlet(flush_block, scale_sum_tasklet, beta, beta_node_new, {}, block.debug_info());
    }

    std::string scaled_input_temp = builder.find_new_name("scaled_input_temp");
    builder.add_container(scaled_input_temp, scalar_type);
    auto& scaled_input_c = builder.add_access(flush_block, scaled_input_temp, block.debug_info());
    builder.add_computational_memlet(flush_block, scale_input_tasklet, "_out", scaled_input_c, {}, block.debug_info());

    auto& flush_add_tasklet = builder.add_tasklet(
        flush_block,
        data_flow::add,
        {"_out", scalar_type},
        {{"_in1", scalar_type}, {"_in2", scalar_type}},
        block.debug_info()
    );
    auto& output_node_new = builder.add_access(flush_block, C_out_var, output_node->debug_info());
    builder.add_computational_memlet(flush_block, scaled_sum_final, flush_add_tasklet, "_in1", {}, block.debug_info());
    builder.add_computational_memlet(flush_block, scaled_input_c, flush_add_tasklet, "_in2", {}, block.debug_info());
    builder
        .add_computational_memlet(flush_block, flush_add_tasklet, "_out", output_node_new, {c_idx}, block.debug_info());


    // Clean up block
    builder.remove_memlet(block, *iedge_a);
    builder.remove_memlet(block, *iedge_b);
    builder.remove_memlet(block, *iedge_c);
    if (alpha_edge) {
        builder.remove_memlet(block, *alpha_edge);
        builder.remove_node(block, *alpha_node);
    }
    if (beta_edge) {
        builder.remove_memlet(block, *beta_edge);
        builder.remove_node(block, *beta_node);
    }
    builder.remove_memlet(block, oedge);
    builder.remove_node(block, *input_node_a);
    builder.remove_node(block, *input_node_b);
    builder.remove_node(block, *input_node_c);
    builder.remove_node(block, *output_node);
    builder.remove_node(block, *this);
    builder.remove_child(parent, block);

    return true;
}

std::unique_ptr<data_flow::DataFlowNode> GEMMNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    auto node_clone = std::unique_ptr<GEMMNode>(new GEMMNode(
        element_id,
        this->debug_info(),
        vertex,
        parent,
        this->implementation_type_,
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
        this->alpha(),
        this->beta()
    ));
    return std::move(node_clone);
}

std::string GEMMNode::toStr() const {
    return LibraryNode::toStr() + "(" + static_cast<char>(precision_) + ", " +
           std::string(BLAS_Layout_to_short_string(layout_)) + ", " + BLAS_Transpose_to_char(trans_a_) +
           BLAS_Transpose_to_char(trans_b_) + ", " + m_->__str__() + ", " + n_->__str__() + ", " + k_->__str__() +
           ", " + lda_->__str__() + ", " + ldb_->__str__() + ", " + ldc_->__str__() + ")";
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

    auto implementation_type = j.at("implementation_type").get<std::string>();

    return builder.add_library_node<GEMMNode>(
        parent, debug_info, implementation_type, precision, layout, trans_a, trans_b, m, n, k, lda, ldb, ldc, alpha, beta
    );
}

GEMMNodeDispatcher_BLAS::GEMMNodeDispatcher_BLAS(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const GEMMNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void GEMMNodeDispatcher_BLAS::dispatch(codegen::PrettyPrinter& stream) {
    stream << "{" << std::endl;
    stream.setIndent(stream.indent() + 4);

    auto& gemm_node = static_cast<const GEMMNode&>(this->node_);

    sdfg::types::Scalar base_type(types::PrimitiveType::Void);
    switch (gemm_node.precision()) {
        case BLAS_Precision::h:
            base_type = types::Scalar(types::PrimitiveType::Half);
            break;
        case BLAS_Precision::s:
            base_type = types::Scalar(types::PrimitiveType::Float);
            break;
        case BLAS_Precision::d:
            base_type = types::Scalar(types::PrimitiveType::Double);
            break;
        default:
            throw std::runtime_error("Invalid BLAS_Precision value");
    }

    auto& graph = this->node_.get_parent();
    for (auto& iedge : graph.in_edges(this->node_)) {
        auto& access_node = static_cast<const data_flow::AccessNode&>(iedge.src());
        std::string name = access_node.data();
        auto& type = this->function_.type(name);

        stream << this->language_extension_.declaration(iedge.dst_conn(), type);
        stream << " = " << name << ";" << std::endl;
    }

    if (std::find(gemm_node.inputs().begin(), gemm_node.inputs().end(), "alpha") == gemm_node.inputs().end()) {
        stream << this->language_extension_.declaration("alpha", base_type);
        stream << " = " << gemm_node.alpha() << ";" << std::endl;
    }
    if (std::find(gemm_node.inputs().begin(), gemm_node.inputs().end(), "beta") == gemm_node.inputs().end()) {
        stream << this->language_extension_.declaration("beta", base_type);
        stream << " = " << gemm_node.beta() << ";" << std::endl;
    }

    stream << "cblas_" << BLAS_Precision_to_string(gemm_node.precision()) << "gemm(";
    stream.setIndent(stream.indent() + 4);
    stream << BLAS_Layout_to_string(gemm_node.layout());
    stream << ", ";
    stream << BLAS_Transpose_to_string(gemm_node.trans_a());
    stream << ", ";
    stream << BLAS_Transpose_to_string(gemm_node.trans_b());
    stream << ", ";
    stream << this->language_extension_.expression(gemm_node.m());
    stream << ", ";
    stream << this->language_extension_.expression(gemm_node.n());
    stream << ", ";
    stream << this->language_extension_.expression(gemm_node.k());
    stream << ", ";
    stream << "alpha";
    stream << ", ";
    stream << "A";
    stream << ", ";
    stream << this->language_extension_.expression(gemm_node.lda());
    stream << ", ";
    stream << "B";
    stream << ", ";
    stream << this->language_extension_.expression(gemm_node.ldb());
    stream << ", ";
    stream << "beta";
    stream << ", ";
    stream << "C";
    stream << ", ";
    stream << this->language_extension_.expression(gemm_node.ldc());

    stream.setIndent(stream.indent() - 4);
    stream << ");" << std::endl;

    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
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
