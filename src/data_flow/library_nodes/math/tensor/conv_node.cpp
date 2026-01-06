#include "sdfg/data_flow/library_nodes/math/tensor/conv_node.h"

#include <map>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/types/type.h"

#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/data_flow/library_nodes/math/blas/gemm_node.h"

namespace sdfg {
namespace math {
namespace tensor {

ConvNode::ConvNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<symbolic::Expression>& kernel_shape,
    const std::vector<symbolic::Expression>& strides,
    const std::vector<symbolic::Expression>& pads,
    const std::vector<symbolic::Expression>& dilations,
    symbolic::Expression group
)
    : TensorNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_Conv,
          {"Y"},
          {"X", "W", "B"}, // X and W are required, B (bias) is optional
          data_flow::ImplementationType_NONE
      ),
      kernel_shape_(kernel_shape), strides_(strides), pads_(pads), dilations_(dilations), group_(group) {}

void ConvNode::validate(const Function& function) const {
    auto& graph = this->get_parent();

    // Custom validation for ConvNode that handles optional bias input
    // We expect X, W as required inputs and optionally B (bias)
    
    // Collect all input edges by connector name
    std::map<std::string, const data_flow::Memlet*> input_edges;
    for (auto& iedge : graph.in_edges(*this)) {
        input_edges[iedge.dst_conn()] = &iedge;
    }

    // Check that required inputs X and W are present
    if (input_edges.find("X") == input_edges.end()) {
        throw InvalidSDFGException("ConvNode: Required input 'X' is not connected");
    }
    if (input_edges.find("W") == input_edges.end()) {
        throw InvalidSDFGException("ConvNode: Required input 'W' is not connected");
    }

    // Validate all connected input memlets are scalar or pointer of scalar
    for (auto& iedge : graph.in_edges(*this)) {
        if (iedge.base_type().type_id() != types::TypeID::Scalar &&
            iedge.base_type().type_id() != types::TypeID::Pointer) {
            throw InvalidSDFGException(
                "ConvNode: Input memlet must be of scalar or pointer type. Found type: " + iedge.base_type().print()
            );
        }
        if (iedge.base_type().type_id() == types::TypeID::Pointer) {
            auto& ptr_type = static_cast<const types::Pointer&>(iedge.base_type());
            if (ptr_type.pointee_type().type_id() != types::TypeID::Scalar) {
                throw InvalidSDFGException(
                    "ConvNode: Input memlet pointer must be flat (pointer to scalar). Found type: " +
                    ptr_type.pointee_type().print()
                );
            }
            if (!iedge.subset().empty()) {
                throw InvalidSDFGException("ConvNode: Input memlet pointer must not be dereferenced.");
            }
        }
    }

    // Validate output memlets are scalar or pointer of scalar
    for (auto& oedge : graph.out_edges(*this)) {
        if (oedge.base_type().type_id() != types::TypeID::Scalar &&
            oedge.base_type().type_id() != types::TypeID::Pointer) {
            throw InvalidSDFGException(
                "ConvNode: Output memlet must be of scalar or pointer type. Found type: " + oedge.base_type().print()
            );
        }
        if (oedge.base_type().type_id() == types::TypeID::Pointer) {
            auto& ptr_type = static_cast<const types::Pointer&>(oedge.base_type());
            if (ptr_type.pointee_type().type_id() != types::TypeID::Scalar) {
                throw InvalidSDFGException(
                    "ConvNode: Output memlet pointer must be flat (pointer to scalar). Found type: " +
                    ptr_type.pointee_type().print()
                );
            }
            if (!oedge.subset().empty()) {
                throw InvalidSDFGException("ConvNode: Output memlet pointer must not be dereferenced.");
            }
        }
    }

    // Validate that all memlets have the same primitive type
    types::PrimitiveType prim_type = this->primitive_type(graph);

    // ConvNode supports floating-point types (check integer support)
    if (!this->supports_integer_types() && types::is_integer(prim_type)) {
        throw InvalidSDFGException(
            "ConvNode: This operation does not support integer types. Found type: " +
            std::string(types::primitive_type_to_string(prim_type))
        );
    }

    // Validate kernel shape is not empty
    if (kernel_shape_.empty()) {
        throw InvalidSDFGException("ConvNode kernel_shape cannot be empty");
    }

    // Validate strides, pads, dilations have consistent dimensions
    size_t spatial_dims = kernel_shape_.size();

    if (!strides_.empty() && strides_.size() != spatial_dims) {
        throw InvalidSDFGException("ConvNode strides must match kernel spatial dimensions");
    }

    if (!pads_.empty() && pads_.size() != 2 * spatial_dims) {
        throw InvalidSDFGException("ConvNode pads must have 2 * spatial dimensions (start and end for each axis)");
    }

    if (!dilations_.empty() && dilations_.size() != spatial_dims) {
        throw InvalidSDFGException("ConvNode dilations must match kernel spatial dimensions");
    }
}

bool ConvNode::expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();

    auto& dataflow = this->get_parent();
    auto& block = static_cast<structured_control_flow::Block&>(*dataflow.get_parent());
    auto& parent = static_cast<structured_control_flow::Sequence&>(*scope_analysis.parent_scope(&block));
    int index = parent.index(block);
    auto& transition = parent.at(index).second;

    // Get primitive type
    auto primitive_type = this->primitive_type(dataflow);
    types::Scalar scalar_type(primitive_type);

    // Get input edges
    auto in_edges = dataflow.in_edges(*this);
    auto in_edges_it = in_edges.begin();

    data_flow::Memlet* x_edge = nullptr;
    data_flow::Memlet* w_edge = nullptr;
    data_flow::Memlet* b_edge = nullptr;

    while (in_edges_it != in_edges.end()) {
        auto& edge = *in_edges_it;
        auto dst_conn = edge.dst_conn();
        if (dst_conn == "X") {
            x_edge = &edge;
        } else if (dst_conn == "W") {
            w_edge = &edge;
        } else if (dst_conn == "B") {
            b_edge = &edge;
        } else {
            throw InvalidSDFGException("ConvNode has unexpected input: " + dst_conn);
        }
        ++in_edges_it;
    }

    if (!x_edge || !w_edge) {
        throw InvalidSDFGException("ConvNode requires X and W inputs");
    }

    auto& y_edge = *dataflow.out_edges(*this).begin();

    // Get access nodes
    auto* x_node = static_cast<data_flow::AccessNode*>(&x_edge->src());
    auto* w_node = static_cast<data_flow::AccessNode*>(&w_edge->src());
    data_flow::AccessNode* b_node = b_edge ? static_cast<data_flow::AccessNode*>(&b_edge->src()) : nullptr;
    auto* y_node = static_cast<data_flow::AccessNode*>(&y_edge.dst());

    // Validate nodes are standalone in the block
    if (!x_node || dataflow.in_degree(*x_node) != 0 || !w_node || dataflow.in_degree(*w_node) != 0 || !y_node ||
        dataflow.out_degree(*y_node) != 0) {
        return false;
    }

    if (b_node && dataflow.in_degree(*b_node) != 0) {
        return false;
    }

    // Check that all other nodes in the block are the expected ones
    for (auto* nd : dataflow.data_nodes()) {
        if (nd != x_node && nd != w_node && nd != y_node && (!b_node || nd != b_node)) {
            return false; // there are other nodes we cannot handle
        }
    }

    // Support n-dimensional convolutions
    size_t spatial_dims = kernel_shape_.size();

    if (spatial_dims == 0) {
        return false; // Need at least 1 spatial dimension
    }

    // Get strides (default to 1 if not provided)
    std::vector<symbolic::Expression> strides_vec;
    for (size_t i = 0; i < spatial_dims; ++i) {
        if (i < strides_.size()) {
            strides_vec.push_back(strides_[i]);
        } else {
            strides_vec.push_back(static_cast<symbolic::Expression>(symbolic::one()));
        }
    }

    // Get padding (default to 0 if not provided)
    // Pads format: [begin_0, begin_1, ..., begin_n, end_0, end_1, ..., end_n]
    std::vector<symbolic::Expression> pads_begin_vec;
    for (size_t i = 0; i < spatial_dims; ++i) {
        if (i < pads_.size()) {
            pads_begin_vec.push_back(pads_[i]);
        } else {
            pads_begin_vec.push_back(static_cast<symbolic::Expression>(symbolic::zero()));
        }
    }

    // Get variable names
    auto& X_var = x_node->data();
    auto& W_var = w_node->data();
    auto& Y_var = y_node->data();

    // Symbolic variables for dimensions (these should be defined based on the problem)
    // For a generic n-dimensional implementation:
    // Input X shape: [N, C_in, D0_in, D1_in, ..., Dn_in]
    // Weight W shape: [C_out, C_in, K0, K1, ..., Kn]
    // Output Y shape: [N, C_out, D0_out, D1_out, ..., Dn_out]

    // Create symbolic dimension variables
    auto N = symbolic::symbol(builder.find_new_name("N"));
    auto C_in = symbolic::symbol(builder.find_new_name("C_in"));
    auto C_out = symbolic::symbol(builder.find_new_name("C_out"));

    // Create symbolic variables for output spatial dimensions
    std::vector<symbolic::Expression> output_spatial_dims;
    for (size_t i = 0; i < spatial_dims; ++i) {
        auto dim_name = builder.find_new_name("D" + std::to_string(i) + "_out");
        output_spatial_dims.push_back(symbolic::symbol(dim_name));
    }

    // Create new sequence for expansion
    auto& new_sequence = builder.add_sequence_before(parent, block, transition.assignments(), block.debug_info());

    // Create nested map structure for convolution:
    // Map over: batch(N), output_channel(C_out), output_height(H_out), output_width(W_out)
    // For loop over: input_channel(C_in), kernel_height(KH), kernel_width(KW)

    structured_control_flow::Sequence* current_scope = &new_sequence;
    std::vector<symbolic::Expression> output_indices;
    std::vector<symbolic::Expression> output_spatial_vars;

    // Map over batch dimension
    std::string n_str = builder.find_new_name("n");
    builder.add_container(n_str, types::Scalar(types::PrimitiveType::UInt64));
    auto n_var = symbolic::symbol(n_str);
    auto& map_n = builder.add_map(
        *current_scope,
        n_var,
        symbolic::Lt(n_var, N),
        symbolic::zero(),
        symbolic::add(n_var, symbolic::one()),
        structured_control_flow::ScheduleType_Sequential::create(),
        {},
        block.debug_info()
    );
    current_scope = &map_n.root();
    output_indices.push_back(n_var);

    // Map over output channel dimension
    std::string oc_str = builder.find_new_name("oc");
    builder.add_container(oc_str, types::Scalar(types::PrimitiveType::UInt64));
    auto oc_var = symbolic::symbol(oc_str);
    auto& map_oc = builder.add_map(
        *current_scope,
        oc_var,
        symbolic::Lt(oc_var, C_out),
        symbolic::zero(),
        symbolic::add(oc_var, symbolic::one()),
        structured_control_flow::ScheduleType_Sequential::create(),
        {},
        block.debug_info()
    );
    current_scope = &map_oc.root();
    output_indices.push_back(oc_var);

    // Map over each output spatial dimension dynamically
    for (size_t i = 0; i < spatial_dims; ++i) {
        std::string od_str = builder.find_new_name("od" + std::to_string(i));
        builder.add_container(od_str, types::Scalar(types::PrimitiveType::UInt64));
        auto od_var = symbolic::symbol(od_str);
        auto& map_od = builder.add_map(
            *current_scope,
            od_var,
            symbolic::Lt(od_var, output_spatial_dims[i]),
            symbolic::zero(),
            symbolic::add(od_var, symbolic::one()),
            structured_control_flow::ScheduleType_Sequential::create(),
            {},
            block.debug_info()
        );
        current_scope = &map_od.root();
        output_indices.push_back(od_var);
        output_spatial_vars.push_back(od_var);
    }

    // Create accumulator variable for the sum
    std::string accum_var = builder.find_new_name("_conv_accum");
    builder.add_container(accum_var, scalar_type);

    // Initialize accumulator to 0
    auto& init_block = builder.add_block(*current_scope, {}, block.debug_info());
    auto& accum_init = builder.add_access(init_block, accum_var, block.debug_info());
    auto& zero_const = builder.add_constant(init_block, "0.0", scalar_type, block.debug_info());
    auto& init_tasklet = builder.add_tasklet(init_block, data_flow::assign, "_out", {"_in"}, block.debug_info());
    builder.add_computational_memlet(init_block, zero_const, init_tasklet, "_in", {}, scalar_type, block.debug_info());
    builder.add_computational_memlet(init_block, init_tasklet, "_out", accum_init, {}, scalar_type, block.debug_info());

    // Create nested for loops for input channels and kernel dimensions
    // For loop over input channels
    std::string ic_str = builder.find_new_name("ic");
    builder.add_container(ic_str, types::Scalar(types::PrimitiveType::UInt64));
    auto ic_var = symbolic::symbol(ic_str);
    auto& for_ic = builder.add_for(
        *current_scope,
        ic_var,
        symbolic::Lt(ic_var, C_in),
        symbolic::zero(),
        symbolic::add(ic_var, symbolic::one()),
        {},
        block.debug_info()
    );
    auto* loop_scope = &for_ic.root();

    // For loops over each kernel spatial dimension
    std::vector<symbolic::Expression> kernel_vars;
    for (size_t i = 0; i < spatial_dims; ++i) {
        std::string k_str = builder.find_new_name("k" + std::to_string(i));
        builder.add_container(k_str, types::Scalar(types::PrimitiveType::UInt64));
        auto k_var = symbolic::symbol(k_str);
        auto& for_k = builder.add_for(
            *loop_scope,
            k_var,
            symbolic::Lt(k_var, kernel_shape_[i]),
            symbolic::zero(),
            symbolic::add(k_var, symbolic::one()),
            {},
            block.debug_info()
        );
        loop_scope = &for_k.root();
        kernel_vars.push_back(k_var);
    }

    // Compute indices for input and weight access
    // Input index: [n, ic, od0 * stride0 - pad0 + k0, od1 * stride1 - pad1 + k1, ...]
    // Weight index: [oc, ic, k0, k1, ...]
    std::vector<symbolic::Expression> input_spatial_indices;
    for (size_t i = 0; i < spatial_dims; ++i) {
        auto input_idx = symbolic::
            add(symbolic::sub(symbolic::mul(output_spatial_vars[i], strides_vec[i]), pads_begin_vec[i]),
                kernel_vars[i]);
        input_spatial_indices.push_back(input_idx);
    }

    // Create computation block
    auto& comp_block = builder.add_block(*loop_scope, {}, block.debug_info());

    // Access input X[n, ic, input_spatial_indices...]
    auto& x_access = builder.add_access(comp_block, X_var, x_node->debug_info());
    // Access weight W[oc, ic, k0, k1, ...]
    auto& w_access = builder.add_access(comp_block, W_var, w_node->debug_info());
    // Access accumulator
    auto& accum_read = builder.add_access(comp_block, accum_var, block.debug_info());
    auto& accum_write = builder.add_access(comp_block, accum_var, block.debug_info());

    // Create FMA tasklet: accum = accum + x * w
    auto& fma_tasklet =
        builder.add_tasklet(comp_block, data_flow::fp_fma, "_out", {"_in1", "_in2", "_in3"}, block.debug_info());

    // Connect edges with proper subsets for n-dimensional convolution
    // X subset: [n, ic, input_spatial_indices...]
    std::vector<symbolic::Expression> x_subset_vec = {n_var, ic_var};
    x_subset_vec.insert(x_subset_vec.end(), input_spatial_indices.begin(), input_spatial_indices.end());

    // W subset: [oc, ic, k0, k1, ...]
    std::vector<symbolic::Expression> w_subset_vec = {oc_var, ic_var};
    w_subset_vec.insert(w_subset_vec.end(), kernel_vars.begin(), kernel_vars.end());

    data_flow::Subset x_subset(x_subset_vec.begin(), x_subset_vec.end());
    data_flow::Subset w_subset(w_subset_vec.begin(), w_subset_vec.end());

    builder.add_computational_memlet(
        comp_block, x_access, fma_tasklet, "_in1", x_subset, x_edge->base_type(), x_edge->debug_info()
    );
    builder.add_computational_memlet(
        comp_block, w_access, fma_tasklet, "_in2", w_subset, w_edge->base_type(), w_edge->debug_info()
    );
    builder.add_computational_memlet(comp_block, accum_read, fma_tasklet, "_in3", {}, scalar_type, block.debug_info());
    builder.add_computational_memlet(comp_block, fma_tasklet, "_out", accum_write, {}, scalar_type, block.debug_info());

    // After all loops, write accumulated result to output (with optional bias)
    auto& output_block = builder.add_block(*current_scope, {}, block.debug_info());
    auto& accum_final = builder.add_access(output_block, accum_var, block.debug_info());
    auto& y_access = builder.add_access(output_block, Y_var, y_node->debug_info());

    data_flow::Subset y_subset(output_indices.begin(), output_indices.end());

    if (b_node) {
        // Add bias: output = accum + bias[oc]
        auto& b_access = builder.add_access(output_block, b_node->data(), b_node->debug_info());
        auto& add_tasklet =
            builder.add_tasklet(output_block, data_flow::fp_add, "_out", {"_in1", "_in2"}, block.debug_info());

        builder.add_computational_memlet(output_block, accum_final, add_tasklet, "_in1", {}, scalar_type, block.debug_info());
        builder.add_computational_memlet(
            output_block, b_access, add_tasklet, "_in2", {oc_var}, b_edge->base_type(), b_edge->debug_info()
        );
        builder.add_computational_memlet(
            output_block, add_tasklet, "_out", y_access, y_subset, y_edge.base_type(), y_edge.debug_info()
        );
    } else {
        // No bias: output = accum
        auto& assign_tasklet =
            builder.add_tasklet(output_block, data_flow::assign, "_out", {"_in"}, block.debug_info());

        builder.add_computational_memlet(output_block, accum_final, assign_tasklet, "_in", {}, scalar_type, block.debug_info());
        builder.add_computational_memlet(
            output_block, assign_tasklet, "_out", y_access, y_subset, y_edge.base_type(), y_edge.debug_info()
        );
    }

    // Clean up the original block
    builder.remove_memlet(block, *x_edge);
    builder.remove_memlet(block, *w_edge);
    if (b_edge) {
        builder.remove_memlet(block, *b_edge);
        builder.remove_node(block, *b_node);
    }
    builder.remove_memlet(block, y_edge);
    builder.remove_node(block, *x_node);
    builder.remove_node(block, *w_node);
    builder.remove_node(block, *y_node);
    builder.remove_node(block, *this);
    builder.remove_child(parent, index + 1);

    return true;
}

symbolic::SymbolSet ConvNode::symbols() const {
    symbolic::SymbolSet syms;

    for (auto& expr : kernel_shape_) {
        for (auto& atom : symbolic::atoms(expr)) {
            syms.insert(atom);
        }
    }
    for (auto& expr : strides_) {
        for (auto& atom : symbolic::atoms(expr)) {
            syms.insert(atom);
        }
    }
    for (auto& expr : pads_) {
        for (auto& atom : symbolic::atoms(expr)) {
            syms.insert(atom);
        }
    }
    for (auto& expr : dilations_) {
        for (auto& atom : symbolic::atoms(expr)) {
            syms.insert(atom);
        }
    }
    for (auto& atom : symbolic::atoms(group_)) {
        syms.insert(atom);
    }

    return syms;
}

void ConvNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    for (auto& expr : kernel_shape_) {
        expr = symbolic::subs(expr, old_expression, new_expression);
    }
    for (auto& expr : strides_) {
        expr = symbolic::subs(expr, old_expression, new_expression);
    }
    for (auto& expr : pads_) {
        expr = symbolic::subs(expr, old_expression, new_expression);
    }
    for (auto& expr : dilations_) {
        expr = symbolic::subs(expr, old_expression, new_expression);
    }
    group_ = symbolic::subs(group_, old_expression, new_expression);
}

std::unique_ptr<data_flow::DataFlowNode> ConvNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(
        new ConvNode(element_id, this->debug_info(), vertex, parent, kernel_shape_, strides_, pads_, dilations_, group_)
    );
}

std::string ConvNode::toStr() const {
    std::string result = "Conv(kernel_shape=[";
    for (size_t i = 0; i < kernel_shape_.size(); ++i) {
        if (i > 0) result += ", ";
        result += kernel_shape_[i]->__str__();
    }
    result += "], strides=[";
    for (size_t i = 0; i < strides_.size(); ++i) {
        if (i > 0) result += ", ";
        result += strides_[i]->__str__();
    }
    result += "], group=" + group_->__str__() + ")";
    return result;
}

nlohmann::json ConvNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const ConvNode& conv_node = static_cast<const ConvNode&>(library_node);
    nlohmann::json j;

    j["code"] = conv_node.code().value();

    serializer::JSONSerializer serializer;

    j["kernel_shape"] = nlohmann::json::array();
    for (auto& dim : conv_node.kernel_shape()) {
        j["kernel_shape"].push_back(serializer.expression(dim));
    }

    j["strides"] = nlohmann::json::array();
    for (auto& stride : conv_node.strides()) {
        j["strides"].push_back(serializer.expression(stride));
    }

    j["pads"] = nlohmann::json::array();
    for (auto& pad : conv_node.pads()) {
        j["pads"].push_back(serializer.expression(pad));
    }

    j["dilations"] = nlohmann::json::array();
    for (auto& dilation : conv_node.dilations()) {
        j["dilations"].push_back(serializer.expression(dilation));
    }

    j["group"] = serializer.expression(conv_node.group());

    return j;
}

data_flow::LibraryNode& ConvNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    assert(j.contains("element_id"));
    assert(j.contains("code"));
    assert(j.contains("debug_info"));
    assert(j.contains("kernel_shape"));

    std::vector<symbolic::Expression> kernel_shape;
    for (const auto& dim : j["kernel_shape"]) {
        kernel_shape.push_back(symbolic::parse(dim.get<std::string>()));
    }

    std::vector<symbolic::Expression> strides;
    if (j.contains("strides")) {
        for (const auto& stride : j["strides"]) {
            strides.push_back(symbolic::parse(stride.get<std::string>()));
        }
    }

    std::vector<symbolic::Expression> pads;
    if (j.contains("pads")) {
        for (const auto& pad : j["pads"]) {
            pads.push_back(symbolic::parse(pad.get<std::string>()));
        }
    }

    std::vector<symbolic::Expression> dilations;
    if (j.contains("dilations")) {
        for (const auto& dilation : j["dilations"]) {
            dilations.push_back(symbolic::parse(dilation.get<std::string>()));
        }
    }

    symbolic::Expression group = symbolic::one();
    if (j.contains("group")) {
        group = symbolic::parse(j["group"].get<std::string>());
    }

    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    return builder.add_library_node<ConvNode>(parent, debug_info, kernel_shape, strides, pads, dilations, group);
}

} // namespace tensor
} // namespace math
} // namespace sdfg
