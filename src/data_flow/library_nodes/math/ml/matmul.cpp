#include "sdfg/data_flow/library_nodes/math/ml/matmul.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

namespace sdfg {
namespace math {
namespace ml {

MatMulNode::MatMulNode(
    size_t element_id, const DebugInfo &debug_info, const graph::Vertex vertex, data_flow::DataFlowGraph &parent
)
    : MathNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_MatMul,
          {"C"},
          {"A", "B"},
          data_flow::ImplementationType_NONE
      ) {}

void MatMulNode::validate(const Function &) const { /* TODO */ }

bool MatMulNode::expand(builder::StructuredSDFGBuilder &builder, analysis::AnalysisManager &analysis_manager) {
    auto &dataflow = this->get_parent();
    auto &block = static_cast<structured_control_flow::Block &>(*dataflow.get_parent());

    auto &scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto &parent = static_cast<structured_control_flow::Sequence &>(*scope_analysis.parent_scope(&block));
    int index = parent.index(block);
    auto &transition = parent.at(index).second;

    // Locate edges
    const data_flow::Memlet *iedge_A = nullptr;
    const data_flow::Memlet *iedge_B = nullptr;
    const data_flow::Memlet *oedge_C = nullptr;
    for (const auto &edge : dataflow.in_edges(*this)) {
        if (edge.dst_conn() == "A") {
            iedge_A = &edge;
        }
        if (edge.dst_conn() == "B") {
            iedge_B = &edge;
        }
    }
    for (const auto &edge : dataflow.out_edges(*this)) {
        if (edge.src_conn() == "C") {
            oedge_C = &edge;
        }
    }
    if (!iedge_A || !iedge_B || !oedge_C) return false;

    std::string A_name = static_cast<const data_flow::AccessNode &>(iedge_A->src()).data();
    std::string B_name = static_cast<const data_flow::AccessNode &>(iedge_B->src()).data();
    std::string C_name = static_cast<const data_flow::AccessNode &>(oedge_C->dst()).data();

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
        oedge_C->end_subset()[0],
        oedge_C->end_subset()[1],
        iedge_A->end_subset()[1],
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
            structured_control_flow::ScheduleType_Sequential::create(),
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
    auto &C_in = builder.add_access(code_block, C_name, block.debug_info());
    auto &C_out = builder.add_access(code_block, C_name, block.debug_info());

    builder.add_computational_memlet(
        code_block, A_in, tasklet, "_in1", {out_syms[0], out_syms[2]}, iedge_A->base_type(), block.debug_info()
    );
    builder.add_computational_memlet(
        code_block, B_in, tasklet, "_in2", {out_syms[1], out_syms[2]}, iedge_B->base_type(), block.debug_info()
    );
    builder.add_computational_memlet(
        code_block, C_in, tasklet, "_in3", {out_syms[0], out_syms[1]}, oedge_C->base_type(), block.debug_info()
    );
    builder.add_computational_memlet(
        code_block, tasklet, "_out", C_out, {out_syms[0], out_syms[1]}, oedge_C->base_type(), block.debug_info()
    );

    // Cleanup old block
    builder.remove_memlet(block, *iedge_A);
    builder.remove_memlet(block, *iedge_B);
    builder.remove_memlet(block, *oedge_C);
    builder.remove_node(block, *this);
    builder.remove_child(parent, index + 1);

    return true;
}

std::unique_ptr<data_flow::DataFlowNode> MatMulNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph &parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(new MatMulNode(element_id, this->debug_info(), vertex, parent));
}

nlohmann::json MatMulNodeSerializer::serialize(const data_flow::LibraryNode &library_node) {
    const MatMulNode &node = static_cast<const MatMulNode &>(library_node);
    nlohmann::json j;

    j["code"] = node.code().value();

    return j;
}

data_flow::LibraryNode &MatMulNodeSerializer::deserialize(
    const nlohmann::json &j, builder::StructuredSDFGBuilder &builder, structured_control_flow::Block &parent
) {
    auto code = j["code"].get<std::string>();
    if (code != LibraryNodeType_MatMul.value()) {
        throw std::runtime_error("Invalid library node code");
    }

    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    return builder.add_library_node<MatMulNode>(parent, debug_info);
}

} // namespace ml
} // namespace math
} // namespace sdfg
