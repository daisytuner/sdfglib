#include "sdfg/data_flow/library_nodes/math/ml/gemm.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

namespace sdfg {
namespace math {
namespace ml {

GemmNode::GemmNode(
    size_t element_id,
    const DebugInfo &debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph &parent,
    const std::string &alpha,
    const std::string &beta,
    bool trans_a,
    bool trans_b
)
    : MathNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_Gemm,
          {"Y"},
          {"A", "B", "C"},
          data_flow::ImplementationType_NONE
      ),
      alpha_(alpha), beta_(beta), trans_a_(trans_a), trans_b_(trans_b) {}

void GemmNode::validate(const Function &) const { /* TODO */ }

bool GemmNode::expand(builder::StructuredSDFGBuilder &builder, analysis::AnalysisManager &analysis_manager) {
    auto &dataflow = this->get_parent();
    auto &block = static_cast<structured_control_flow::Block &>(*dataflow.get_parent());

    auto &scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto &parent = static_cast<structured_control_flow::Sequence &>(*scope_analysis.parent_scope(&block));
    int index = parent.index(block);
    auto &transition = parent.at(index).second;

    // Locate edges
    const data_flow::Memlet *iedge_A = nullptr;
    const data_flow::Memlet *iedge_B = nullptr;
    const data_flow::Memlet *iedge_C = nullptr;
    const data_flow::Memlet *oedge_Y = nullptr;
    for (const auto &edge : dataflow.in_edges(*this)) {
        if (edge.dst_conn() == "A") {
            iedge_A = &edge;
        }
        if (edge.dst_conn() == "B") {
            iedge_B = &edge;
        }
        if (edge.dst_conn() == "C") {
            iedge_C = &edge;
        }
    }
    for (const auto &edge : dataflow.out_edges(*this)) {
        if (edge.src_conn() == "Y") {
            oedge_Y = &edge;
        }
    }
    if (!iedge_A || !iedge_B || !oedge_Y) return false;

    bool has_C_in = iedge_C != nullptr;

    std::string A_name = static_cast<const data_flow::AccessNode &>(iedge_A->src()).data();
    std::string B_name = static_cast<const data_flow::AccessNode &>(iedge_B->src()).data();
    std::string C_in_name = has_C_in ? static_cast<const data_flow::AccessNode &>(iedge_C->src()).data() : "";
    std::string C_out_name = static_cast<const data_flow::AccessNode &>(oedge_Y->dst()).data();

    // Create new sequence before
    auto &new_sequence = builder.add_sequence_before(parent, block, transition.assignments(), block.debug_info());
    structured_control_flow::Sequence *last_scope = &new_sequence;

    // Create maps over output subset dims (parallel dims)
    data_flow::Subset domain_begin = {
        symbolic::integer(0),
        symbolic::integer(0),
        symbolic::integer(0),
    };
    data_flow::Subset domain_end = {
        oedge_Y->end_subset()[0],
        oedge_Y->end_subset()[1],
        trans_a_ ? iedge_A->end_subset()[1] : iedge_A->end_subset()[0],
    };

    std::vector<symbolic::Expression> out_syms;
    structured_control_flow::Map *last_map = nullptr;
    for (size_t d = 0; d < domain_begin.size(); ++d) {
        std::string indvar_str = builder.find_new_name("_i");
        builder.add_container(indvar_str, types::Scalar(types::PrimitiveType::UInt64));
        auto indvar = symbolic::symbol(indvar_str);
        auto init = domain_begin[d];
        auto update = symbolic::add(indvar, symbolic::one());
        auto cond = symbolic::Lt(indvar, symbolic::add(domain_end[d], symbolic::one()));
        last_map = &builder.add_map(
            *last_scope,
            indvar,
            cond,
            init,
            update,
            structured_control_flow::ScheduleType_Sequential,
            {},
            block.debug_info()
        );
        last_scope = &last_map->root();
        out_syms.push_back(indvar);
    }

    // Create innermost block
    auto &code_block = builder.add_block(*last_scope);
    auto &tasklet =
        builder
            .add_tasklet(code_block, data_flow::TaskletCode::fma, "_out", {"_in1", "_in2", "_in3"}, block.debug_info());

    auto &A_in = builder.add_access(code_block, A_name, block.debug_info());
    auto &B_in = builder.add_access(code_block, B_name, block.debug_info());
    auto &C_in = builder.add_access(code_block, has_C_in ? C_in_name : C_out_name, block.debug_info());
    auto &C_out = builder.add_access(code_block, C_out_name, block.debug_info());

    data_flow::Subset subset_A;
    if (trans_a_) {
        subset_A = {out_syms[1], out_syms[0]};
    } else {
        subset_A = {out_syms[0], out_syms[1]};
    }
    data_flow::Subset subset_B;
    if (trans_b_) {
        subset_B = {out_syms[1], out_syms[0]};
    } else {
        subset_B = {out_syms[0], out_syms[1]};
    }
    data_flow::Subset subset_C = {out_syms[0], out_syms[1]};

    builder
        .add_computational_memlet(code_block, A_in, tasklet, "_in1", subset_A, iedge_A->base_type(), block.debug_info());
    builder
        .add_computational_memlet(code_block, B_in, tasklet, "_in2", subset_B, iedge_B->base_type(), block.debug_info());
    builder
        .add_computational_memlet(code_block, C_in, tasklet, "_in3", subset_C, oedge_Y->base_type(), block.debug_info());
    builder
        .add_computational_memlet(code_block, tasklet, "_out", C_out, subset_C, oedge_Y->base_type(), block.debug_info());

    // Cleanup old block
    builder.remove_memlet(block, *iedge_A);
    builder.remove_memlet(block, *iedge_B);
    if (has_C_in) {
        builder.remove_memlet(block, *iedge_C);
    }
    builder.remove_memlet(block, *oedge_Y);
    builder.remove_node(block, *this);
    builder.remove_child(parent, index + 1);

    return true;
}

std::unique_ptr<data_flow::DataFlowNode> GemmNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph &parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(
        new GemmNode(element_id, this->debug_info(), vertex, parent, alpha_, beta_, trans_a_, trans_b_)
    );
}

nlohmann::json GemmNodeSerializer::serialize(const data_flow::LibraryNode &library_node) {
    const GemmNode &node = static_cast<const GemmNode &>(library_node);
    nlohmann::json j;

    j["code"] = node.code().value();
    j["alpha"] = node.alpha();
    j["beta"] = node.beta();
    j["trans_a"] = node.trans_a();
    j["trans_b"] = node.trans_b();

    return j;
}

data_flow::LibraryNode &GemmNodeSerializer::deserialize(
    const nlohmann::json &j, builder::StructuredSDFGBuilder &builder, structured_control_flow::Block &parent
) {
    auto code = j["code"].get<std::string>();
    if (code != LibraryNodeType_Gemm.value()) {
        throw std::runtime_error("Invalid library node code");
    }

    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    auto alpha = j["alpha"].get<std::string>();
    auto beta = j["beta"].get<std::string>();
    auto trans_a = j["trans_a"].get<bool>();
    auto trans_b = j["trans_b"].get<bool>();

    return builder.add_library_node<GemmNode>(parent, debug_info, alpha, beta, trans_a, trans_b);
}

} // namespace ml
} // namespace math
} // namespace sdfg
