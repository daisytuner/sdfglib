#include "sdfg/data_flow/library_nodes/math/tensor/conv_node.h"

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
          {"X", "W"},  // B (bias) is optional; if provided as an edge, it must use connector "B"
          data_flow::ImplementationType_NONE
      ),
      kernel_shape_(kernel_shape),
      strides_(strides),
      pads_(pads),
      dilations_(dilations),
      group_(group) {}

void ConvNode::validate(const Function& function) const {
    TensorNode::validate(function);

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
    if (!x_node || dataflow.in_degree(*x_node) != 0 ||
        !w_node || dataflow.in_degree(*w_node) != 0 ||
        !y_node || dataflow.out_degree(*y_node) != 0) {
        return false;
    }

    if (b_node && dataflow.in_degree(*b_node) != 0) {
        return false;
    }

    // For simplicity in this initial implementation, we'll expand a basic 2D convolution
    // The im2col transformation creates a matrix where each column contains the flattened
    // values from one receptive field of the input
    
    // Get variable names (for future implementation)
    // auto& X_var = x_node->data();
    // auto& W_var = w_node->data();
    // auto& Y_var = y_node->data();

    // Full im2col + GEMM expansion not yet implemented
    // Future implementation will:
    // 1. Create im2col transformation to convert input patches into columns
    // 2. Reshape weights for matrix multiplication
    // 3. Create GEMMNode for efficient matrix multiplication
    // 4. Add bias if present
    // 5. Reshape output to final tensor shape
    
    return false;
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

std::unique_ptr<data_flow::DataFlowNode>
ConvNode::clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
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
