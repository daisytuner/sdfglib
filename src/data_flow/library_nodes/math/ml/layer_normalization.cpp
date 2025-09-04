#include "sdfg/data_flow/library_nodes/math/ml/layer_normalization.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

namespace sdfg {
namespace math {
namespace ml {

LayerNormalizationNode::LayerNormalizationNode(
    size_t element_id,
    const DebugInfo &debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph &parent,
    int axis,
    const std::string &epsilon
)
    : MathNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_LayerNormalization,
          {"Y"},
          {"X", "Scale", "B"},
          data_flow::ImplementationType_NONE
      ),
      axis_(axis), epsilon_(epsilon) {}

void LayerNormalizationNode::validate(const Function &) const { /* TODO */ }

bool LayerNormalizationNode::expand(builder::StructuredSDFGBuilder &builder, analysis::AnalysisManager &analysis_manager) {
    auto &dataflow = this->get_parent();
    auto &block = static_cast<structured_control_flow::Block &>(*dataflow.get_parent());

    auto &scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto &parent = static_cast<structured_control_flow::Sequence &>(*scope_analysis.parent_scope(&block));
    int index = parent.index(block);
    auto &transition = parent.at(index).second;

    // Locate edges
    const data_flow::Memlet *iedge_input = nullptr;
    const data_flow::Memlet *iedge_scale = nullptr;
    const data_flow::Memlet *iedge_bias = nullptr;
    const data_flow::Memlet *oedge_output = nullptr;
    for (const auto &edge : dataflow.in_edges(*this)) {
        if (edge.dst_conn() == "X") {
            iedge_input = &edge;
        } else if (edge.dst_conn() == "Scale") {
            iedge_scale = &edge;
        } else if (edge.dst_conn() == "B") {
            iedge_bias = &edge;
        }
    }
    for (const auto &edge : dataflow.out_edges(*this)) {
        if (edge.src_conn() == "Y") {
            oedge_output = &edge;
        }
    }
    if (!iedge_input || !iedge_scale || !oedge_output) return false;

    std::string input_name = static_cast<const data_flow::AccessNode &>(iedge_input->src()).data();
    std::string scale_name = static_cast<const data_flow::AccessNode &>(iedge_scale->src()).data();
    std::string bias_name;
    if (iedge_bias) {
        bias_name = static_cast<const data_flow::AccessNode &>(iedge_bias->src()).data();
    }
    std::string output_name = static_cast<const data_flow::AccessNode &>(oedge_output->dst()).data();

    // Create new sequence before
    auto &new_sequence = builder.add_sequence_before(parent, block, transition.assignments(), block.debug_info());
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
            structured_control_flow::ScheduleType_Sequential(),
            {},
            block.debug_info()
        );
        last_scope = &last_map->root();
        loop_syms.push_back(indvar);
    }

    // Initialize temp variables for mean and variance
    std::string mean_temp_name = builder.find_new_name("_mean_tmp");
    std::string var_temp_name = builder.find_new_name("_var_tmp");
    std::string count_temp_name = builder.find_new_name("_count_tmp");
    types::Scalar temp_type(types::PrimitiveType::Float);
    builder.add_container(mean_temp_name, temp_type);
    builder.add_container(var_temp_name, temp_type);
    builder.add_container(count_temp_name, temp_type);

    auto &init_block = builder.add_block(*last_scope);
    auto &init_mean_tasklet = builder.add_tasklet(init_block, data_flow::TaskletCode::assign, "_out", {"0.0f"});
    auto &init_var_tasklet = builder.add_tasklet(init_block, data_flow::TaskletCode::assign, "_out", {"0.0f"});
    auto &init_count_tasklet = builder.add_tasklet(init_block, data_flow::TaskletCode::assign, "_out", {"0.0f"});
    auto &mean_access_init = builder.add_access(init_block, mean_temp_name);
    auto &var_access_init = builder.add_access(init_block, var_temp_name);
    auto &count_access_init = builder.add_access(init_block, count_temp_name);
    builder.add_computational_memlet(init_block, init_mean_tasklet, "_out", mean_access_init, {}, temp_type);
    builder.add_computational_memlet(init_block, init_var_tasklet, "_out", var_access_init, {}, temp_type);
    builder.add_computational_memlet(init_block, init_count_tasklet, "_out", count_access_init, {}, temp_type);

    // Add reduction for loop to compute mean and variance
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
    auto red_map = &builder.add_for(*last_scope, red_indvar, red_cond, red_init, red_update, {}, block.debug_info());

    // Create innermost block for mean and variance computation
    auto &compute_block = builder.add_block(red_map->root());

    // Create access nodes
    auto &input_access = builder.add_access(compute_block, input_name);
    auto &mean_access_in = builder.add_access(compute_block, mean_temp_name);
    auto &mean_access_out = builder.add_access(compute_block, mean_temp_name);
    auto &var_access_in = builder.add_access(compute_block, var_temp_name);
    auto &var_access_out = builder.add_access(compute_block, var_temp_name);
    auto &count_access_in = builder.add_access(compute_block, count_temp_name);
    auto &count_access_out = builder.add_access(compute_block, count_temp_name);

    // Create index expressions for input
    std::vector<symbolic::Expression> input_subset = loop_syms;

    // Replace the reduction axis index with the reduction variable for input
    if (axis_ >= 0 && axis_ < static_cast<int>(input_subset.size())) {
        input_subset.insert(input_subset.begin() + axis_, red_indvar);
    } else if (axis_ < 0) {
        input_subset.push_back(red_indvar);
    }

    // Add to mean (reduction)
    auto &add_mean_tasklet = builder.add_tasklet(compute_block, data_flow::TaskletCode::add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(
        compute_block, input_access, add_mean_tasklet, "_in1", input_subset, iedge_input->base_type()
    );
    builder.add_computational_memlet(compute_block, mean_access_in, add_mean_tasklet, "_in2", {}, temp_type);
    builder.add_computational_memlet(compute_block, add_mean_tasklet, "_out", mean_access_out, {}, temp_type);

    // Add to variance (reduction of squared values)
    auto &square_tasklet = builder.add_tasklet(compute_block, data_flow::TaskletCode::mul, "_out", {"_in1", "_in2"});
    auto &square_temp_access = builder.add_access(compute_block, builder.find_new_name("_square_tmp"));
    builder.add_computational_memlet(
        compute_block, input_access, square_tasklet, "_in1", input_subset, iedge_input->base_type()
    );
    builder.add_computational_memlet(
        compute_block, input_access, square_tasklet, "_in2", input_subset, iedge_input->base_type()
    );
    builder.add_computational_memlet(compute_block, square_tasklet, "_out", square_temp_access, {}, temp_type);

    auto &add_var_tasklet = builder.add_tasklet(compute_block, data_flow::TaskletCode::add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(compute_block, square_temp_access, add_var_tasklet, "_in1", {}, temp_type);
    builder.add_computational_memlet(compute_block, var_access_in, add_var_tasklet, "_in2", {}, temp_type);
    builder.add_computational_memlet(compute_block, add_var_tasklet, "_out", var_access_out, {}, temp_type);

    // Increment count
    auto &add_count_tasklet = builder.add_tasklet(compute_block, data_flow::TaskletCode::add, "_out", {"_in1", "1.0f"});
    builder.add_computational_memlet(compute_block, count_access_in, add_count_tasklet, "_in1", {}, temp_type);
    builder.add_computational_memlet(compute_block, add_count_tasklet, "_out", count_access_out, {}, temp_type);

    // Create normalization block
    auto &norm_block = builder.add_block(*last_scope);

    // Create access nodes for normalization
    auto &mean_access_norm = builder.add_access(norm_block, mean_temp_name);
    auto &var_access_norm = builder.add_access(norm_block, var_temp_name);
    auto &count_access_norm = builder.add_access(norm_block, count_temp_name);
    auto &input_access_norm = builder.add_access(norm_block, input_name);
    auto &scale_access_norm = builder.add_access(norm_block, scale_name);
    auto &output_access_norm = builder.add_access(norm_block, output_name);

    // Compute mean by dividing sum by count
    auto &div_mean_tasklet = builder.add_tasklet(norm_block, data_flow::TaskletCode::div, "_out", {"_in1", "_in2"});
    auto &mean_result_access = builder.add_access(norm_block, builder.find_new_name("_mean_result"));
    builder.add_computational_memlet(norm_block, mean_access_norm, div_mean_tasklet, "_in1", {}, temp_type);
    builder.add_computational_memlet(norm_block, count_access_norm, div_mean_tasklet, "_in2", {}, temp_type);
    builder.add_computational_memlet(norm_block, div_mean_tasklet, "_out", mean_result_access, {}, temp_type);

    // Compute variance by dividing sum of squares by count and subtracting mean squared
    auto &div_var_tasklet = builder.add_tasklet(norm_block, data_flow::TaskletCode::div, "_out", {"_in1", "_in2"});
    auto &var_div_access = builder.add_access(norm_block, builder.find_new_name("_var_div"));
    builder.add_computational_memlet(norm_block, var_access_norm, div_var_tasklet, "_in1", {}, temp_type);
    builder.add_computational_memlet(norm_block, count_access_norm, div_var_tasklet, "_in2", {}, temp_type);
    builder.add_computational_memlet(norm_block, div_var_tasklet, "_out", var_div_access, {}, temp_type);

    auto &square_mean_tasklet = builder.add_tasklet(norm_block, data_flow::TaskletCode::mul, "_out", {"_in", "_in"});
    auto &mean_squared_access = builder.add_access(norm_block, builder.find_new_name("_mean_squared"));
    builder.add_computational_memlet(norm_block, mean_result_access, square_mean_tasklet, "_in", {}, temp_type);
    builder.add_computational_memlet(norm_block, square_mean_tasklet, "_out", mean_squared_access, {}, temp_type);

    auto &sub_var_tasklet = builder.add_tasklet(norm_block, data_flow::TaskletCode::sub, "_out", {"_in1", "_in2"});
    auto &var_result_access = builder.add_access(norm_block, builder.find_new_name("_var_result"));
    builder.add_computational_memlet(norm_block, var_div_access, sub_var_tasklet, "_in1", {}, temp_type);
    builder.add_computational_memlet(norm_block, mean_squared_access, sub_var_tasklet, "_in2", {}, temp_type);
    builder.add_computational_memlet(norm_block, sub_var_tasklet, "_out", var_result_access, {}, temp_type);

    // Add epsilon to variance and compute standard deviation
    auto &add_epsilon_tasklet =
        builder.add_tasklet(norm_block, data_flow::TaskletCode::add, "_out", {"_in1", epsilon_});
    auto &var_eps_access = builder.add_access(norm_block, builder.find_new_name("_var_eps"));
    builder.add_computational_memlet(norm_block, var_result_access, add_epsilon_tasklet, "_in1", {}, temp_type);
    builder.add_computational_memlet(norm_block, add_epsilon_tasklet, "_out", var_eps_access, {}, temp_type);

    auto &sqrt_tasklet = builder.add_tasklet(norm_block, data_flow::TaskletCode::sqrt, "_out", {"_in"});
    auto &std_dev_access = builder.add_access(norm_block, builder.find_new_name("_std_dev"));
    builder.add_computational_memlet(norm_block, var_eps_access, sqrt_tasklet, "_in", {}, temp_type);
    builder.add_computational_memlet(norm_block, sqrt_tasklet, "_out", std_dev_access, {}, temp_type);

    // Normalize: (x - mean) / std_dev
    auto &sub_norm_tasklet = builder.add_tasklet(norm_block, data_flow::TaskletCode::sub, "_out", {"_in1", "_in2"});
    auto &centered_access = builder.add_access(norm_block, builder.find_new_name("_centered"));
    builder.add_computational_memlet(
        norm_block, input_access_norm, sub_norm_tasklet, "_in1", loop_syms, iedge_input->base_type()
    );
    builder.add_computational_memlet(norm_block, mean_result_access, sub_norm_tasklet, "_in2", {}, temp_type);
    builder.add_computational_memlet(norm_block, sub_norm_tasklet, "_out", centered_access, {}, temp_type);

    auto &div_norm_tasklet = builder.add_tasklet(norm_block, data_flow::TaskletCode::div, "_out", {"_in1", "_in2"});
    auto &normalized_access = builder.add_access(norm_block, builder.find_new_name("_normalized"));
    builder.add_computational_memlet(norm_block, centered_access, div_norm_tasklet, "_in1", {}, temp_type);
    builder.add_computational_memlet(norm_block, std_dev_access, div_norm_tasklet, "_in2", {}, temp_type);
    builder.add_computational_memlet(norm_block, div_norm_tasklet, "_out", normalized_access, {}, temp_type);

    // Apply scale and bias: scale * normalized + bias (if bias is provided)
    auto &mul_scale_tasklet = builder.add_tasklet(norm_block, data_flow::TaskletCode::mul, "_out", {"_in1", "_in2"});
    auto &scaled_access = builder.add_access(norm_block, builder.find_new_name("_scaled"));
    builder.add_computational_memlet(norm_block, normalized_access, mul_scale_tasklet, "_in1", {}, temp_type);
    builder.add_computational_memlet(
        norm_block, scale_access_norm, mul_scale_tasklet, "_in2", loop_syms, iedge_scale->base_type()
    );
    builder.add_computational_memlet(norm_block, mul_scale_tasklet, "_out", scaled_access, {}, temp_type);

    if (iedge_bias) {
        // Add bias if provided
        auto &bias_access_norm = builder.add_access(norm_block, bias_name);
        auto &add_bias_tasklet = builder.add_tasklet(norm_block, data_flow::TaskletCode::add, "_out", {"_in1", "_in2"});
        builder.add_computational_memlet(norm_block, scaled_access, add_bias_tasklet, "_in1", {}, temp_type);
        builder.add_computational_memlet(
            norm_block, bias_access_norm, add_bias_tasklet, "_in2", loop_syms, iedge_bias->base_type()
        );
        builder.add_computational_memlet(
            norm_block, add_bias_tasklet, "_out", output_access_norm, loop_syms, oedge_output->base_type()
        );
    } else {
        // No bias, just assign scaled result to output
        auto &assign_tasklet = builder.add_tasklet(norm_block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(norm_block, scaled_access, assign_tasklet, "_in", {}, temp_type);
        builder.add_computational_memlet(
            norm_block, assign_tasklet, "_out", output_access_norm, loop_syms, oedge_output->base_type()
        );
    }

    // Cleanup old block
    builder.remove_memlet(block, *iedge_input);
    builder.remove_memlet(block, *iedge_scale);
    if (iedge_bias) {
        builder.remove_memlet(block, *iedge_bias);
    }
    builder.remove_memlet(block, *oedge_output);
    builder.remove_node(block, *this);
    builder.remove_child(parent, index + 1);

    return true;
}

std::unique_ptr<data_flow::DataFlowNode> LayerNormalizationNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph &parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(
        new LayerNormalizationNode(element_id, this->debug_info(), vertex, parent, axis_, epsilon_)
    );
}

nlohmann::json LayerNormalizationNodeSerializer::serialize(const data_flow::LibraryNode &library_node) {
    const LayerNormalizationNode &node = static_cast<const LayerNormalizationNode &>(library_node);
    nlohmann::json j;

    j["code"] = node.code().value();
    j["axis"] = node.axis();
    j["epsilon"] = node.epsilon();

    return j;
}

data_flow::LibraryNode &LayerNormalizationNodeSerializer::deserialize(
    const nlohmann::json &j, builder::StructuredSDFGBuilder &builder, structured_control_flow::Block &parent
) {
    auto code = j["code"].get<std::string>();
    if (code != LibraryNodeType_LayerNormalization.value()) {
        throw std::runtime_error("Invalid library node code");
    }

    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    auto axis = j["axis"].get<int>();
    auto epsilon = j["epsilon"].get<std::string>();

    return builder.add_library_node<LayerNormalizationNode>(parent, debug_info, axis, epsilon);
}

} // namespace ml
} // namespace math
} // namespace sdfg
