#include "sdfg/data_flow/library_nodes/math/ml/softmax.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

namespace sdfg {
namespace math {
namespace ml {

SoftmaxNode::SoftmaxNode(
    size_t element_id,
    const DebugInfo &debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph &parent,
    int axis
)
    : MathNode(
          element_id, debug_info, vertex, parent, LibraryNodeType_Softmax, {"output"}, {"input"}, data_flow::ImplementationType_NONE
      ),
      axis_(axis) {}

void SoftmaxNode::validate(const Function &) const { /* TODO */ }

bool SoftmaxNode::expand(builder::StructuredSDFGBuilder &builder, analysis::AnalysisManager &analysis_manager) {
    auto &dataflow = this->get_parent();
    auto &block = static_cast<structured_control_flow::Block &>(*dataflow.get_parent());

    auto &scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto &parent = static_cast<structured_control_flow::Sequence &>(*scope_analysis.parent_scope(&block));

    // Locate edges
    const data_flow::Memlet *iedge_input = nullptr;
    const data_flow::Memlet *oedge_output = nullptr;
    for (const auto &edge : dataflow.in_edges(*this)) {
        if (edge.dst_conn() == "input") {
            iedge_input = &edge;
        }
    }
    for (const auto &edge : dataflow.out_edges(*this)) {
        if (edge.src_conn() == "output") {
            oedge_output = &edge;
        }
    }
    if (!iedge_input || !oedge_output) return false;

    std::string input_name = static_cast<const data_flow::AccessNode &>(iedge_input->src()).data();
    std::string output_name = static_cast<const data_flow::AccessNode &>(oedge_output->dst()).data();

    // Create new sequence before
    auto &new_sequence = builder.add_sequence_before(parent, block, block.debug_info()).first;
    structured_control_flow::Sequence *last_scope = &new_sequence;

    // Create maps over output subset dims (parallel dims)
    data_flow::Subset domain_begin = oedge_output->begin_subset();
    data_flow::Subset domain_end = oedge_output->end_subset();

    std::vector<symbolic::Expression> loop_syms;
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
        loop_syms.push_back(indvar);
    }
    
    // Initialize temp variable to zero
    std::string temp_name = builder.find_new_name("_tmp");
    std::string temp_name2 = builder.find_new_name("_tmp");
    types::Scalar temp_type(types::PrimitiveType::Float);
    builder.add_container(temp_name, temp_type);
    builder.add_container(temp_name2, temp_type);
    
    auto &init_block = builder.add_block(*last_scope);
    auto &init_tasklet = builder.add_tasklet(init_block, data_flow::TaskletCode::assign, "_out", {"0.0f"});
    auto &tmp_access_init = builder.add_access(init_block, temp_name);
    builder.add_computational_memlet(init_block, init_tasklet, "_out", tmp_access_init, {}, temp_type);

    // add reduction for loop
    symbolic::Expression red_begin;
    symbolic::Expression red_end;
    if (axis_ >= 0) {
        red_begin = iedge_input->begin_subset()[axis_];
        red_end = iedge_input->end_subset()[axis_];
    } else {
        red_begin = iedge_input->begin_subset().back();
        red_end = iedge_input->end_subset().back();
    }
    std::string red_name = builder.find_new_name("_i");
    builder.add_container(red_name, types::Scalar(types::PrimitiveType::UInt64));
    auto red_indvar = symbolic::symbol(red_name);
    auto red_init = red_begin;
    auto red_update = symbolic::add(red_indvar, symbolic::one());
    auto red_cond = symbolic::Lt(red_indvar, symbolic::add(red_end, symbolic::one()));
    auto red_map = &builder.add_for(
        *last_scope,
        red_indvar,
        red_cond,
        red_init,
        red_update,
        {},
        block.debug_info()
    );

    // Create innermost block
    auto &code_block = builder.add_block(red_map->root());
    
    // Create access nodes for input and output
    auto &input_access = builder.add_access(code_block, input_name);
    auto &tmp2_access = builder.add_access(code_block, temp_name2);
    auto &tmp_access_out = builder.add_access(code_block, temp_name);
    auto &tmp_access_in = builder.add_access(code_block, temp_name2);
    
    // Create index expressions for input and output
    std::vector<symbolic::Expression> input_subset = loop_syms;
    
    // Replace the reduction axis index with the reduction variable for input
    if (axis_ >= 0 && axis_ < static_cast<int>(input_subset.size())) {
        input_subset.insert(input_subset.begin() + axis_, red_indvar);
    } else if (axis_ < 0) {
        input_subset.push_back(red_indvar);
    }
    
    // Compute exponential
    auto &exp_tasklet = builder.add_tasklet(code_block, data_flow::TaskletCode::expf, "_out", {"_in"});
    builder.add_computational_memlet(code_block, input_access, exp_tasklet, "_in", input_subset, iedge_input->base_type());
    builder.add_computational_memlet(code_block, exp_tasklet, "_out", tmp2_access, {}, temp_type);
    
    // Add to temp (reduction)
    auto &add_tasklet = builder.add_tasklet(code_block, data_flow::TaskletCode::add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(code_block, tmp_access_in, add_tasklet, "_in1", {}, temp_type);
    builder.add_computational_memlet(code_block, tmp2_access, add_tasklet, "_in2", {}, temp_type);
    builder.add_computational_memlet(code_block, add_tasklet, "_out", tmp_access_out, {}, temp_type);

    // Create writeback - assign the accumulated sum to output
    auto &writeback_block = builder.add_block(*last_scope);
    auto &tmp_access_wb = builder.add_access(writeback_block, temp_name);
    auto &output_access_wb = builder.add_access(writeback_block, output_name);
    auto &writeback_tasklet = builder.add_tasklet(writeback_block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(writeback_block, tmp_access_wb, writeback_tasklet, "_in", {}, temp_type);
    builder.add_computational_memlet(writeback_block, writeback_tasklet, "_out", output_access_wb, loop_syms, oedge_output->base_type());

    // Cleanup old block
    builder.remove_memlet(block, *iedge_input);
    builder.remove_memlet(block, *oedge_output);
    builder.remove_node(block, *this);
    builder.remove_child(parent, block);

    return true;
}

std::unique_ptr<data_flow::DataFlowNode> SoftmaxNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph &parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(new SoftmaxNode(
        element_id, this->debug_info(), vertex, parent, axis_
    ));
}

nlohmann::json SoftmaxNodeSerializer::serialize(const data_flow::LibraryNode &library_node) {
    const SoftmaxNode &node = static_cast<const SoftmaxNode &>(library_node);
    nlohmann::json j;

    j["code"] = node.code().value();
    j["axis"] = node.axis();

    return j;
}

data_flow::LibraryNode &SoftmaxNodeSerializer::deserialize(
    const nlohmann::json &j, builder::StructuredSDFGBuilder &builder, structured_control_flow::Block &parent
) {
    auto code = j["code"].get<std::string>();
    if (code != LibraryNodeType_Softmax.value()) {
        throw std::runtime_error("Invalid library node code");
    }

    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    auto axis = j["axis"].get<int>();

    return builder.add_library_node<SoftmaxNode>(parent, debug_info, axis);
}

} // namespace ml
} // namespace math
} // namespace sdfg
