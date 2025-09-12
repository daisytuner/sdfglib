#include "sdfg/data_flow/library_nodes/math/ml/batch_normalization.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

namespace sdfg {
namespace math {
namespace ml {

BatchNormalizationNode::BatchNormalizationNode(
    size_t element_id,
    const DebugInfo &debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph &parent,
    const std::vector<symbolic::Expression> &shape,
    int axis,
    const std::string &epsilon
)
    : MathNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_BatchNormalization,
          {"Y"},
          {"X", "Scale", "B", "input_mean", "input_var"},
          data_flow::ImplementationType_NONE
      ),
      shape_(shape), axis_(axis), epsilon_(epsilon) {}

symbolic::SymbolSet BatchNormalizationNode::symbols() const {
    symbolic::SymbolSet syms;
    for (const auto &dim : shape_) {
        for (auto &atom : symbolic::atoms(dim)) {
            syms.insert(atom);
        }
    }
    return syms;
}

void BatchNormalizationNode::
    replace(const symbolic::Expression &old_expression, const symbolic::Expression &new_expression) {
    for (auto &dim : shape_) {
        dim = symbolic::subs(dim, old_expression, new_expression);
    }
}

void BatchNormalizationNode::validate(const Function &) const {}

bool BatchNormalizationNode::expand(builder::StructuredSDFGBuilder &builder, analysis::AnalysisManager &analysis_manager) {
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
    const data_flow::Memlet *iedge_mean = nullptr;
    const data_flow::Memlet *iedge_var = nullptr;
    const data_flow::Memlet *oedge_output = nullptr;
    for (const auto &edge : dataflow.in_edges(*this)) {
        if (edge.dst_conn() == "X") {
            iedge_input = &edge;
        } else if (edge.dst_conn() == "Scale") {
            iedge_scale = &edge;
        } else if (edge.dst_conn() == "B") {
            iedge_bias = &edge;
        } else if (edge.dst_conn() == "input_mean") {
            iedge_mean = &edge;
        } else if (edge.dst_conn() == "input_var") {
            iedge_var = &edge;
        }
    }
    for (const auto &edge : dataflow.out_edges(*this)) {
        if (edge.src_conn() == "Y") {
            oedge_output = &edge;
        }
    }
    if (!iedge_input || !iedge_scale || !iedge_bias || !iedge_mean || !iedge_var || !oedge_output) return false;

    std::string input_name = static_cast<const data_flow::AccessNode &>(iedge_input->src()).data();
    std::string scale_name = static_cast<const data_flow::AccessNode &>(iedge_scale->src()).data();
    std::string bias_name = static_cast<const data_flow::AccessNode &>(iedge_bias->src()).data();
    std::string mean_name = static_cast<const data_flow::AccessNode &>(iedge_mean->src()).data();
    std::string var_name = static_cast<const data_flow::AccessNode &>(iedge_var->src()).data();
    std::string output_name = static_cast<const data_flow::AccessNode &>(oedge_output->dst()).data();

    // Create new sequence before
    auto &new_sequence = builder.add_sequence_before(parent, block, transition.assignments(), block.debug_info());
    structured_control_flow::Sequence *last_scope = &new_sequence;

    std::vector<symbolic::Expression> loop_syms;
    structured_control_flow::Map *last_map = nullptr;
    for (size_t d = 0; d < this->shape_.size(); ++d) {
        std::string indvar_str = builder.find_new_name("_i");
        builder.add_container(indvar_str, types::Scalar(types::PrimitiveType::UInt64));
        auto indvar = symbolic::symbol(indvar_str);
        auto init = symbolic::zero();
        auto update = symbolic::add(indvar, symbolic::one());
        auto cond = symbolic::Lt(indvar, this->shape_[d]);
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
        loop_syms.push_back(indvar);
    }

    // Create normalization block
    auto &norm_block = builder.add_block(*last_scope);

    // Create access nodes for normalization
    auto &input_access_norm = builder.add_access(norm_block, input_name);
    auto &scale_access_norm = builder.add_access(norm_block, scale_name);
    auto &bias_access_norm = builder.add_access(norm_block, bias_name);
    auto &mean_access_norm = builder.add_access(norm_block, mean_name);
    auto &var_access_norm = builder.add_access(norm_block, var_name);
    auto &output_access_norm = builder.add_access(norm_block, output_name);

    // Add epsilon to variance and compute standard deviation
    auto &add_epsilon_tasklet =
        builder.add_tasklet(norm_block, data_flow::TaskletCode::add, "_out", {"_in1", epsilon_});
    auto &var_eps_access = builder.add_access(norm_block, builder.find_new_name("_var_eps"));
    builder.add_computational_memlet(
        norm_block, var_access_norm, add_epsilon_tasklet, "_in1", loop_syms, iedge_var->base_type()
    );
    builder
        .add_computational_memlet(norm_block, add_epsilon_tasklet, "_out", var_eps_access, {}, iedge_var->base_type());

    auto &sqrt_tasklet = builder.add_tasklet(norm_block, data_flow::TaskletCode::sqrt, "_out", {"_in"});
    auto &std_dev_access = builder.add_access(norm_block, builder.find_new_name("_std_dev"));
    builder.add_computational_memlet(norm_block, var_eps_access, sqrt_tasklet, "_in", {}, iedge_var->base_type());
    builder.add_computational_memlet(norm_block, sqrt_tasklet, "_out", std_dev_access, {}, iedge_var->base_type());

    // Normalize: (x - mean) / std_dev
    auto &sub_norm_tasklet = builder.add_tasklet(norm_block, data_flow::TaskletCode::sub, "_out", {"_in1", "_in2"});
    auto &centered_access = builder.add_access(norm_block, builder.find_new_name("_centered"));
    builder.add_computational_memlet(
        norm_block, input_access_norm, sub_norm_tasklet, "_in1", loop_syms, iedge_input->base_type()
    );
    builder.add_computational_memlet(
        norm_block, mean_access_norm, sub_norm_tasklet, "_in2", loop_syms, iedge_mean->base_type()
    );
    builder
        .add_computational_memlet(norm_block, sub_norm_tasklet, "_out", centered_access, {}, iedge_input->base_type());

    auto &div_norm_tasklet = builder.add_tasklet(norm_block, data_flow::TaskletCode::div, "_out", {"_in1", "_in2"});
    auto &normalized_access = builder.add_access(norm_block, builder.find_new_name("_normalized"));
    builder
        .add_computational_memlet(norm_block, centered_access, div_norm_tasklet, "_in1", {}, iedge_input->base_type());
    builder
        .add_computational_memlet(norm_block, std_dev_access, div_norm_tasklet, "_in2", loop_syms, iedge_var->base_type());
    builder
        .add_computational_memlet(norm_block, div_norm_tasklet, "_out", normalized_access, {}, iedge_input->base_type());

    // Apply scale and bias: scale * normalized + bias
    auto &mul_scale_tasklet = builder.add_tasklet(norm_block, data_flow::TaskletCode::mul, "_out", {"_in1", "_in2"});
    auto &scaled_access = builder.add_access(norm_block, builder.find_new_name("_scaled"));
    builder
        .add_computational_memlet(norm_block, normalized_access, mul_scale_tasklet, "_in1", {}, iedge_input->base_type());
    builder.add_computational_memlet(
        norm_block, scale_access_norm, mul_scale_tasklet, "_in2", loop_syms, iedge_scale->base_type()
    );
    builder.add_computational_memlet(norm_block, mul_scale_tasklet, "_out", scaled_access, {}, iedge_input->base_type());

    auto &add_bias_tasklet = builder.add_tasklet(norm_block, data_flow::TaskletCode::add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(norm_block, scaled_access, add_bias_tasklet, "_in1", {}, iedge_input->base_type());
    builder.add_computational_memlet(
        norm_block, bias_access_norm, add_bias_tasklet, "_in2", loop_syms, iedge_bias->base_type()
    );
    builder.add_computational_memlet(
        norm_block, add_bias_tasklet, "_out", output_access_norm, loop_syms, oedge_output->base_type()
    );

    // Cleanup old block
    builder.remove_memlet(block, *iedge_input);
    builder.remove_memlet(block, *iedge_scale);
    if (iedge_bias) {
        builder.remove_memlet(block, *iedge_bias);
    }
    builder.remove_memlet(block, *iedge_mean);
    builder.remove_memlet(block, *iedge_var);
    builder.remove_memlet(block, *oedge_output);
    builder.remove_node(block, *this);
    builder.remove_child(parent, index + 1);

    return true;
}

std::unique_ptr<data_flow::DataFlowNode> BatchNormalizationNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph &parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(
        new BatchNormalizationNode(element_id, this->debug_info(), vertex, parent, this->shape_, axis_, epsilon_)
    );
}

nlohmann::json BatchNormalizationNodeSerializer::serialize(const data_flow::LibraryNode &library_node) {
    const BatchNormalizationNode &node = static_cast<const BatchNormalizationNode &>(library_node);
    nlohmann::json j;

    j["code"] = node.code().value();
    j["axis"] = node.axis();
    j["epsilon"] = node.epsilon();

    serializer::JSONSerializer serializer;
    j["shape"] = nlohmann::json::array();
    for (auto &dim : node.shape()) {
        j["shape"].push_back(serializer.expression(dim));
    }

    return j;
}

data_flow::LibraryNode &BatchNormalizationNodeSerializer::deserialize(
    const nlohmann::json &j, builder::StructuredSDFGBuilder &builder, structured_control_flow::Block &parent
) {
    auto code = j["code"].get<std::string>();
    if (code != LibraryNodeType_BatchNormalization.value()) {
        throw std::runtime_error("Invalid library node code");
    }

    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    std::vector<symbolic::Expression> shape;
    for (const auto &dim : j["shape"]) {
        shape.push_back(symbolic::parse(dim.get<std::string>()));
    }

    auto axis = j["axis"].get<int>();
    auto epsilon = j["epsilon"].get<std::string>();

    return builder.add_library_node<BatchNormalizationNode>(parent, debug_info, shape, axis, epsilon);
}

} // namespace ml
} // namespace math
} // namespace sdfg
