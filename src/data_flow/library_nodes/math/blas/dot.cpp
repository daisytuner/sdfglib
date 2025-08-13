#include "sdfg/data_flow/library_nodes/math/blas/dot.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

#include "sdfg/analysis/scope_analysis.h"

namespace sdfg {
namespace math {
namespace blas {

DotNode::DotNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const data_flow::ImplementationType& implementation_type,
    const BLAS_Precision& precision,
    symbolic::Expression n,
    symbolic::Expression incx,
    symbolic::Expression incy
)
    : MathNode(element_id, debug_info, vertex, parent, LibraryNodeType_DOT, {"res"}, {"x", "y"}, implementation_type),
      precision_(precision), n_(n), incx_(incx), incy_(incy) {}

BLAS_Precision DotNode::precision() const { return this->precision_; };

symbolic::Expression DotNode::n() const { return this->n_; };

symbolic::Expression DotNode::incx() const { return this->incx_; };

symbolic::Expression DotNode::incy() const { return this->incy_; };

void DotNode::validate(const Function& function) const {
    auto& graph = this->get_parent();

    if (graph.in_degree(*this) != this->inputs_.size()) {
        throw InvalidSDFGException("DotNode must have " + std::to_string(this->inputs_.size()) + " inputs");
    }
    if (graph.out_degree(*this) != 1) {
        throw InvalidSDFGException("DotNode must have 1 output");
    }

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
            throw InvalidSDFGException("DotNode input " + input + " not found");
        }
    }

    auto& oedge = *graph.out_edges(*this).begin();
    if (oedge.src_conn() != this->outputs_.at(0)) {
        throw InvalidSDFGException("DotNode output " + this->outputs_.at(0) + " not found");
    }

    auto& x_memlet = memlets.at("x");
    auto& x_subset_begin = x_memlet->begin_subset();
    auto& x_subset_end = x_memlet->end_subset();
    if (x_subset_begin.size() != 1) {
        throw InvalidSDFGException("DotNode input x must have 1 dimensions");
    }

    auto& y_memlet = memlets.at("y");
    auto& y_subset_begin = y_memlet->begin_subset();
    auto& y_subset_end = y_memlet->end_subset();
    if (y_subset_begin.size() != 1) {
        throw InvalidSDFGException("DotNode input y must have 1 dimensions");
    }
}

bool DotNode::expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& dataflow = this->get_parent();
    auto& block = static_cast<structured_control_flow::Block&>(*dataflow.get_parent());

    auto& scope_analyisis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto& parent = static_cast<structured_control_flow::Sequence&>(*scope_analyisis.parent_scope(&block));

    const data_flow::Memlet* iedge_x = nullptr;
    const data_flow::Memlet* iedge_y = nullptr;
    for (const auto& iedge : dataflow.in_edges(*this)) {
        if (iedge.dst_conn() == "x") {
            iedge_x = &iedge;
        } else if (iedge.dst_conn() == "y") {
            iedge_y = &iedge;
        }
    }

    const data_flow::Memlet* oedge_res = nullptr;
    for (const auto& oedge : dataflow.out_edges(*this)) {
        if (oedge.src_conn() == "res") {
            oedge_res = &oedge;
            break;
        }
    }

    // Check if legal
    auto& input_node_x = static_cast<const data_flow::AccessNode&>(iedge_x->src());
    auto& input_node_y = static_cast<const data_flow::AccessNode&>(iedge_y->src());
    auto& output_node_res = static_cast<const data_flow::AccessNode&>(oedge_res->dst());
    if (dataflow.in_degree(input_node_x) != 0 || dataflow.in_degree(input_node_y) != 0 ||
        dataflow.out_degree(output_node_res) != 0) {
        return false;
    }

    auto& new_sequence = builder.add_sequence_before(parent, block, block.debug_info()).first;

    std::string loop_var = builder.find_new_name("_i");
    builder.add_container(loop_var, types::Scalar(types::PrimitiveType::UInt64));

    auto loop_indvar = symbolic::symbol(loop_var);
    auto loop_init = symbolic::integer(0);
    auto loop_condition = symbolic::Lt(loop_indvar, this->n_);
    auto loop_update = symbolic::add(loop_indvar, symbolic::integer(1));

    auto& loop =
        builder.add_for(new_sequence, loop_indvar, loop_condition, loop_init, loop_update, {}, block.debug_info());
    auto& body = loop.root();

    auto& new_block = builder.add_block(body);

    auto& res_in = builder.add_access(new_block, output_node_res.data());
    auto& res_out = builder.add_access(new_block, output_node_res.data());
    auto& x = builder.add_access(new_block, input_node_x.data());
    auto& y = builder.add_access(new_block, input_node_y.data());

    auto& tasklet = builder.add_tasklet(new_block, data_flow::TaskletCode::fma, "_out", {"_in1", "_in2", "_in3"});

    builder.add_computational_memlet(
        new_block,
        x,
        tasklet,
        "_in1",
        {symbolic::mul(loop_indvar, this->incx_)},
        iedge_x->base_type(),
        iedge_x->debug_info()
    );
    builder.add_computational_memlet(
        new_block,
        y,
        tasklet,
        "_in2",
        {symbolic::mul(loop_indvar, this->incy_)},
        iedge_y->base_type(),
        iedge_y->debug_info()
    );
    builder
        .add_computational_memlet(new_block, res_in, tasklet, "_in3", {}, oedge_res->base_type(), oedge_res->debug_info());
    builder.add_computational_memlet(
        new_block, tasklet, "_out", res_out, {}, oedge_res->base_type(), oedge_res->debug_info()
    );

    // Clean up
    builder.remove_memlet(block, *iedge_x);
    builder.remove_memlet(block, *iedge_y);
    builder.remove_memlet(block, *oedge_res);
    builder.remove_node(block, input_node_x);
    builder.remove_node(block, input_node_y);
    builder.remove_node(block, output_node_res);
    builder.remove_node(block, *this);
    builder.remove_child(parent, block);

    return true;
}

std::unique_ptr<data_flow::DataFlowNode> DotNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    auto node_clone = std::unique_ptr<DotNode>(new DotNode(
        element_id,
        this->debug_info(),
        vertex,
        parent,
        this->implementation_type_,
        this->precision_,
        this->n_,
        this->incx_,
        this->incy_
    ));
    return std::move(node_clone);
}

nlohmann::json DotNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const DotNode& gemm_node = static_cast<const DotNode&>(library_node);
    nlohmann::json j;

    serializer::JSONSerializer serializer;
    j["code"] = gemm_node.code().value();
    j["precision"] = gemm_node.precision();
    j["n"] = serializer.expression(gemm_node.n());
    j["incx"] = serializer.expression(gemm_node.incx());
    j["incy"] = serializer.expression(gemm_node.incy());

    return j;
}

data_flow::LibraryNode& DotNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    // Assertions for required fields
    assert(j.contains("element_id"));
    assert(j.contains("code"));
    assert(j.contains("debug_info"));

    auto code = j["code"].get<std::string>();
    if (code != LibraryNodeType_DOT.value()) {
        throw std::runtime_error("Invalid library node code");
    }

    // Extract debug info using JSONSerializer
    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    auto precision = j.at("precision").get<BLAS_Precision>();
    auto n = SymEngine::Expression(j.at("n"));
    auto incx = SymEngine::Expression(j.at("incx"));
    auto incy = SymEngine::Expression(j.at("incy"));

    auto implementation_type = j.at("implementation_type").get<std::string>();

    return builder.add_library_node<DotNode>(parent, debug_info, implementation_type, precision, n, incx, incy);
}

DotNodeDispatcher_BLAS::DotNodeDispatcher_BLAS(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const DotNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void DotNodeDispatcher_BLAS::dispatch(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    stream << "{" << std::endl;
    stream.setIndent(stream.indent() + 4);

    auto& dot_node = static_cast<const DotNode&>(this->node_);

    sdfg::types::Scalar base_type(types::PrimitiveType::Void);
    switch (dot_node.precision()) {
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
    for (auto& oedge : graph.out_edges(this->node_)) {
        auto& access_node = static_cast<const data_flow::AccessNode&>(oedge.dst());
        std::string name = access_node.data();
        auto& type = this->function_.type(name);

        stream << this->language_extension_.declaration(oedge.src_conn(), type);
        stream << ";" << std::endl;
    }

    std::string res_name = this->node_.outputs().at(0);
    stream << res_name << " = ";
    stream << "cblas_" << BLAS_Precision_to_string(dot_node.precision()) << "dot(";
    stream.setIndent(stream.indent() + 4);
    stream << this->language_extension_.expression(dot_node.n());
    stream << ", ";
    stream << "x";
    stream << ", ";
    stream << this->language_extension_.expression(dot_node.incx());
    stream << ", ";
    stream << "y";
    stream << ", ";
    stream << this->language_extension_.expression(dot_node.incy());
    stream.setIndent(stream.indent() - 4);
    stream << ");" << std::endl;

    for (auto& oedge : graph.out_edges(this->node_)) {
        auto& access_node = static_cast<const data_flow::AccessNode&>(oedge.dst());
        std::string name = access_node.data();
        auto& type = this->function_.type(name);
        stream << name << " = " << oedge.src_conn() << ";" << std::endl;
    }

    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
}

DotNodeDispatcher_CUBLAS::DotNodeDispatcher_CUBLAS(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const DotNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void DotNodeDispatcher_CUBLAS::dispatch(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    throw std::runtime_error("DotNodeDispatcher_CUBLAS not implemented");
}

} // namespace blas
} // namespace math
} // namespace sdfg
