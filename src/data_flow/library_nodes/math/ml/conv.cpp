#include "sdfg/data_flow/library_nodes/math/ml/conv.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

#include "sdfg/analysis/scope_analysis.h"

namespace sdfg {
namespace math {
namespace ml {

ConvNode::ConvNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    bool has_bias,
    std::vector<size_t> dilations,
    std::vector<size_t> kernel_shape,
    std::vector<size_t> pads,
    std::vector<size_t> strides
)
    : MathNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_Conv,
          {"Y"},
          {"X", "W"},
          data_flow::ImplementationType_NONE
      ),
      has_bias_(has_bias), dilations_(dilations), kernel_shape_(kernel_shape), pads_(pads), strides_(strides) {
    if (has_bias_) {
        this->inputs_.push_back("B");
    }
}

void ConvNode::validate(const Function& function) const {
    // TODO: Implement
}

bool ConvNode::has_bias() const { return has_bias_; }

std::vector<size_t> ConvNode::dilations() const { return dilations_; }

std::vector<size_t> ConvNode::kernel_shape() const { return kernel_shape_; }

std::vector<size_t> ConvNode::pads() const { return pads_; }

std::vector<size_t> ConvNode::strides() const { return strides_; }

bool ConvNode::expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = builder.subject();
    auto& dataflow = this->get_parent();
    auto& block = static_cast<structured_control_flow::Block&>(*dataflow.get_parent());

    auto& scope_analyisis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto& parent = static_cast<structured_control_flow::Sequence&>(*scope_analyisis.parent_scope(&block));

    const data_flow::Memlet* iedge_X = nullptr;
    const data_flow::Memlet* iedge_W = nullptr;
    const data_flow::Memlet* iedge_B = nullptr;
    for (const auto& iedge : dataflow.in_edges(*this)) {
        if (iedge.dst_conn() == "X") {
            iedge_X = &iedge;
        } else if (iedge.dst_conn() == "W") {
            iedge_W = &iedge;
        } else if (iedge.dst_conn() == "B") {
            iedge_B = &iedge;
        }
    }

    const data_flow::Memlet* oedge_Y = nullptr;
    for (const auto& oedge : dataflow.out_edges(*this)) {
        if (oedge.src_conn() == "Y") {
            oedge_Y = &oedge;
            break;
        }
    }

    data_flow::Subset dims_X = iedge_X->end_subset();
    data_flow::Subset dims_W = iedge_W->end_subset();
    data_flow::Subset dims_B;
    if (iedge_B != nullptr) {
        dims_B = iedge_B->end_subset();
    }
    data_flow::Subset dims_Y = oedge_Y->end_subset();

    auto& new_sequence = builder.add_sequence_before(parent, block, block.debug_info()).first;
    structured_control_flow::Sequence* last_scope = &new_sequence;

    /************************
     * Parallel dimensions *
     ************************/
    // Generate one Map per parallel dimension of the output tensor (Y).
    const auto& begin_Y_subset = oedge_Y->begin_subset();
    const auto& end_Y_subset = oedge_Y->end_subset();

    data_flow::Subset out_subset;
    std::vector<symbolic::Expression> parallel_syms;
    structured_control_flow::Map* last_map = nullptr;
    for (size_t dim = 0; dim < begin_Y_subset.size(); ++dim) {
        const auto& dim_begin = begin_Y_subset[dim];
        const auto& dim_end = end_Y_subset[dim];

        std::string indvar_str = builder.find_new_name("_i");
        builder.add_container(indvar_str, types::Scalar(types::PrimitiveType::UInt64));

        auto indvar = symbolic::symbol(indvar_str);
        auto init = dim_begin;
        auto update = symbolic::add(indvar, symbolic::one());
        auto condition = symbolic::Lt(indvar, symbolic::add(dim_end, symbolic::one()));

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
        out_subset.push_back(indvar);
        parallel_syms.push_back(indvar);
    }

    /************************
     * Reduction dimensions *
     ************************/
    // For convolution, we reduce over input channels and kernel dimensions.
    // Assuming weight tensor layout (M, C, k1, k2, ...), skip the first dim (output channels).
    const auto& begin_W_subset = iedge_W->begin_subset();
    const auto& end_W_subset = iedge_W->end_subset();

    std::vector<symbolic::Expression> reduction_syms;
    structured_control_flow::For* last_for = nullptr;
    for (size_t dim = 1; dim < begin_W_subset.size(); ++dim) {
        const auto& dim_begin = begin_W_subset[dim];
        const auto& dim_end = end_W_subset[dim];

        std::string indvar_str = builder.find_new_name("_r");
        builder.add_container(indvar_str, types::Scalar(types::PrimitiveType::UInt64));

        auto indvar = symbolic::symbol(indvar_str);
        auto init = dim_begin;
        auto update = symbolic::add(indvar, symbolic::one());
        auto condition = symbolic::Lt(indvar, symbolic::add(dim_end, symbolic::one()));

        last_for = &builder.add_for(*last_scope, indvar, condition, init, update, {}, block.debug_info());
        last_scope = &last_for->root();
        reduction_syms.push_back(indvar);
    }

    // Add innermost code block – convolution computation.
    auto& code_block = builder.add_block(*last_scope, {}, block.debug_info());

    // Determine scalar element type from output container.
    const auto& output_container = this->outputs_.at(0);
    const auto& output_type = sdfg.type(output_container);
    types::Scalar scalar_type(output_type.primitive_type());

    // Reuse debug infos from original access nodes (if available).
    const DebugInfo& dbg_X = iedge_X->src().debug_info();
    const DebugInfo& dbg_W = iedge_W->src().debug_info();
    const DebugInfo& dbg_Y = oedge_Y->dst().debug_info();
    const DebugInfo dbg_B = (iedge_B != nullptr) ? iedge_B->src().debug_info() : DebugInfo();

    // Find names of input and output containers
    std::string X_name = static_cast<const data_flow::AccessNode&>(iedge_X->src()).data();
    std::string W_name = static_cast<const data_flow::AccessNode&>(iedge_W->src()).data();
    std::string Y_name = static_cast<const data_flow::AccessNode&>(oedge_Y->dst()).data();

    // Create new access nodes inside the innermost block.
    auto& X_acc = builder.add_access(code_block, X_name, dbg_X);
    auto& W_acc = builder.add_access(code_block, W_name, dbg_W);
    auto& Y_acc_in = builder.add_access(code_block, Y_name, dbg_Y);
    auto& Y_acc_out = builder.add_access(code_block, Y_name, dbg_Y);
    // Bias handled after reduction loops; no need to access B inside the reduction tasklet.

    /********************
     * Build subsets    *
     ********************/
    // Helper lambdas to safely fetch stride/dilation/pad values.
    auto int_expr = [](size_t v) { return symbolic::integer(static_cast<int64_t>(v)); };

    auto get_stride = [&](size_t idx) -> symbolic::Expression {
        if (idx < strides_.size()) {
            return int_expr(strides_[idx]);
        }
        return symbolic::one();
    };

    auto get_dilation = [&](size_t idx) -> symbolic::Expression {
        if (idx < dilations_.size()) {
            return int_expr(dilations_[idx]);
        }
        return symbolic::one();
    };

    auto get_pad = [&](size_t idx) -> symbolic::Expression {
        if (idx < pads_.size()) {
            return int_expr(pads_[idx]);
        }
        return symbolic::zero();
    };

    const size_t spatial_dims = kernel_shape_.size();

    // Extract commonly-used indices.
    auto get_parallel_sym = [&](size_t idx) -> symbolic::Expression {
        if (idx < parallel_syms.size()) return parallel_syms[idx];
        return symbolic::zero();
    };

    auto get_reduction_sym = [&](size_t idx) -> symbolic::Expression {
        if (idx < reduction_syms.size()) return reduction_syms[idx];
        return symbolic::zero();
    };

    auto N_idx = get_parallel_sym(0);
    auto M_idx = get_parallel_sym(1);

    // Input channel and kernel indices come from reduction variables.
    auto C_idx = get_reduction_sym(0);

    // Build X subset.
    data_flow::Subset subset_X;
    subset_X.push_back(N_idx); // Batch dim
    subset_X.push_back(C_idx); // Input channel dim
    for (size_t d = 0; d < spatial_dims; ++d) {
        symbolic::Expression out_d = get_parallel_sym(2 + d);
        symbolic::Expression ker_d = get_reduction_sym(1 + d);

        auto in_d = symbolic::
            sub(symbolic::add(symbolic::mul(out_d, get_stride(d)), symbolic::mul(ker_d, get_dilation(d))), get_pad(d));
        subset_X.push_back(in_d);
    }

    // Build W subset.
    data_flow::Subset subset_W;
    subset_W.push_back(M_idx); // Output channel (filter)
    subset_W.push_back(C_idx); // Input channel
    for (size_t d = 0; d < spatial_dims; ++d) {
        symbolic::Expression ker_d = get_reduction_sym(1 + d);
        subset_W.push_back(ker_d);
    }

    // Output Y subset is simply the parallel indices computed earlier.
    data_flow::Subset subset_Y = out_subset;

    // Bias subset (only along output channels).
    data_flow::Subset subset_B;
    if (has_bias_) {
        subset_B.push_back(M_idx);
    }

    /************************
     * Add computation node *
     ************************/
    // Create tasklet performing fused-multiply-add: _out = _x * _w + _y
    std::vector<std::pair<std::string, types::Scalar>> t_inputs{
        {"_x", scalar_type}, {"_w", scalar_type}, {"_y", scalar_type}
    };
    if (has_bias_) {
        // Bias will be added after reduction, so no change here.
    }

    auto& tasklet =
        builder
            .add_tasklet(code_block, data_flow::TaskletCode::fma, {"_out", scalar_type}, t_inputs, block.debug_info());

    // Connect memlets.
    builder.add_computational_memlet(code_block, X_acc, tasklet, "_x", subset_X, block.debug_info());
    builder.add_computational_memlet(code_block, W_acc, tasklet, "_w", subset_W, block.debug_info());
    builder.add_computational_memlet(code_block, Y_acc_in, tasklet, "_y", subset_Y, block.debug_info());
    builder.add_computational_memlet(code_block, tasklet, "_out", Y_acc_out, subset_Y, block.debug_info());

    // Bias: add once per output element outside reduction loops.
    if (has_bias_) {
        // Insert after the reduction loops (i.e., right after they finish).
        // We add a single tasklet in the parent scope (last_map root).
        std::string B_name = static_cast<const data_flow::AccessNode&>(iedge_B->src()).data();
        auto& bias_block = builder.add_block(new_sequence, {}, block.debug_info());
        auto& B_acc_local = builder.add_access(bias_block, B_name, dbg_B);
        auto& Y_acc2_in = builder.add_access(bias_block, Y_name, dbg_Y);
        auto& Y_acc2_out = builder.add_access(bias_block, Y_name, dbg_Y);

        auto& bias_tasklet = builder.add_tasklet(
            bias_block,
            data_flow::TaskletCode::add,
            {"_out", scalar_type},
            {{"_bias", scalar_type}, {"_y", scalar_type}},
            block.debug_info()
        );
        builder.add_computational_memlet(bias_block, B_acc_local, bias_tasklet, "_bias", subset_B, block.debug_info());
        builder.add_computational_memlet(bias_block, Y_acc2_in, bias_tasklet, "_y", subset_Y, block.debug_info());
        builder.add_computational_memlet(bias_block, bias_tasklet, "_out", Y_acc2_out, subset_Y, block.debug_info());
    }

    // Clean up block
    builder.remove_memlet(block, *iedge_X);
    builder.remove_memlet(block, *iedge_W);
    if (iedge_B != nullptr) {
        builder.remove_memlet(block, *iedge_B);
    }
    builder.remove_memlet(block, *oedge_Y);
    builder.remove_node(block, *this);
    builder.remove_child(parent, block);

    return true;
}

std::unique_ptr<data_flow::DataFlowNode> ConvNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(new ConvNode(
        element_id,
        this->debug_info(),
        vertex,
        parent,
        this->has_bias_,
        this->dilations_,
        this->kernel_shape_,
        this->pads_,
        this->strides_
    ));
}

nlohmann::json ConvNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const ConvNode& relu_node = static_cast<const ConvNode&>(library_node);
    nlohmann::json j;

    j["code"] = relu_node.code().value();
    j["outputs"] = relu_node.outputs();
    j["inputs"] = relu_node.inputs();
    j["has_bias"] = relu_node.has_bias();
    j["dilations"] = relu_node.dilations();
    j["kernel_shape"] = relu_node.kernel_shape();
    j["pads"] = relu_node.pads();
    j["strides"] = relu_node.strides();

    return j;
}

data_flow::LibraryNode& ConvNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    auto code = j["code"].get<std::string>();
    if (code != LibraryNodeType_Conv.value()) {
        throw std::runtime_error("Invalid library node code");
    }

    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    auto outputs = j.at("outputs").get<std::vector<std::string>>();
    auto inputs = j.at("inputs").get<std::vector<std::string>>();
    auto has_bias = j.at("has_bias").get<bool>();
    auto dilations = j.at("dilations").get<std::vector<size_t>>();
    auto kernel_shape = j.at("kernel_shape").get<std::vector<size_t>>();
    auto pads = j.at("pads").get<std::vector<size_t>>();
    auto strides = j.at("strides").get<std::vector<size_t>>();

    return builder.add_library_node<ConvNode>(parent, debug_info, has_bias, dilations, kernel_shape, pads, strides);
}

} // namespace ml
} // namespace math
} // namespace sdfg
