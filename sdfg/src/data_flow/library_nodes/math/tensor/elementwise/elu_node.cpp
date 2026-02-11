#include "sdfg/data_flow/library_nodes/math/tensor/elementwise/elu_node.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise/elementwise_node.h"

namespace sdfg {
namespace math {
namespace tensor {

EluNode::EluNode(
    size_t element_id, const DebugInfo& debug_info, const graph::Vertex vertex, data_flow::DataFlowGraph& parent
)
    : TensorNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_Elu,
          {"_out"},
          {"_in1"},
          data_flow::ImplementationType_NONE
      ) {}

symbolic::SymbolSet EluNode::symbols() const { return {}; }

void EluNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {}

bool EluNode::expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& dataflow = this->get_parent();
    auto& block = static_cast<structured_control_flow::Block&>(*dataflow.get_parent());
    if (dataflow.in_degree(*this) != 1 || dataflow.out_degree(*this) != 1) {
        return false;
    }

    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto& parent = static_cast<structured_control_flow::Sequence&>(*scope_analysis.parent_scope(&block));
    int index = parent.index(block);
    auto& transition = parent.at(index).second;

    auto& input = this->inputs_.at(0);
    auto& output = this->outputs_.at(0);

    auto& iedge = *dataflow.in_edges(*this).begin();
    auto& oedge = *dataflow.out_edges(*this).begin();

    // Checks if legal
    auto& input_node = static_cast<data_flow::AccessNode&>(iedge.src());
    auto& output_node = static_cast<data_flow::AccessNode&>(oedge.dst());
    if (dataflow.in_degree(input_node) != 0 || dataflow.out_degree(output_node) != 0) {
        return false;
    }
    types::Scalar element_type(oedge.base_type().primitive_type());

    // Add new graph after the current block
    auto& code_block = builder.add_block_before(parent, block, transition.assignments(), block.debug_info());

    // exp(x)
    auto& input_node_new = builder.add_access(code_block, input_node.data());
    auto& output_node_exp = builder.add_access(code_block, output_node.data());
    auto& exp_tensor_node = builder.add_library_node<math::tensor::ElementwiseNode>(
        code_block, code_block.debug_info(), math::cmath::CMathFunction::exp, oedge.base_type().primitive_type()
    );
    builder
        .add_computational_memlet(code_block, input_node_new, exp_tensor_node, "_in1", iedge.subset(), iedge.base_type());
    builder
        .add_computational_memlet(code_block, exp_tensor_node, "_out", output_node_exp, oedge.subset(), oedge.base_type());

    // exp(x) - 1
    auto& output_node_sub = builder.add_access(code_block, output_node.data());
    auto& one_node = builder.add_constant(code_block, "1.0", element_type);
    auto& sub_tensor_node = builder.add_library_node<
        math::tensor::ElementwiseNode>(code_block, code_block.debug_info(), data_flow::TaskletCode::fp_sub);
    builder
        .add_computational_memlet(code_block, output_node_exp, sub_tensor_node, "_in1", oedge.subset(), oedge.base_type());
    builder.add_computational_memlet(code_block, one_node, sub_tensor_node, "_in2", {}, element_type);
    builder
        .add_computational_memlet(code_block, sub_tensor_node, "_out", output_node_sub, oedge.subset(), oedge.base_type());


    // Clean up block
    builder.remove_memlet(block, iedge);
    builder.remove_memlet(block, oedge);
    builder.remove_node(block, input_node);
    builder.remove_node(block, output_node);
    builder.remove_node(block, *this);
    builder.remove_child(parent, index + 1);

    return true;
}

void EluNode::validate(const Function& function) const {
    auto& graph = this->get_parent();

    if (graph.in_degree(*this) != 1) {
        throw InvalidSDFGException("EluNode must have exactly one input memlet");
    }
    if (graph.out_degree(*this) != 1) {
        throw InvalidSDFGException("EluNode must have exactly one output memlet");
    }

    auto& iedge = *graph.in_edges(*this).begin();
    auto& oedge = *graph.out_edges(*this).begin();
    if (iedge.base_type().type_id() != types::TypeID::Tensor) {
        throw InvalidSDFGException("EluNode input must be a tensor");
    }
    if (oedge.base_type().type_id() != types::TypeID::Tensor) {
        throw InvalidSDFGException("EluNode output must be a tensor");
    }
}

std::unique_ptr<data_flow::DataFlowNode> EluNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(new EluNode(element_id, this->debug_info(), vertex, parent));
}

nlohmann::json EluNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const EluNode& elem_node = static_cast<const EluNode&>(library_node);
    nlohmann::json j;

    j["code"] = elem_node.code().value();

    return j;
}

data_flow::LibraryNode& EluNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    // Assertions for required fields
    assert(j.contains("element_id"));
    assert(j.contains("code"));
    assert(j.contains("debug_info"));

    // Extract debug info using JSONSerializer
    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    auto code = j["code"].get<std::string>();
    return static_cast<EluNode&>(builder.add_library_node<EluNode>(parent, debug_info));
}

} // namespace tensor
} // namespace math
} // namespace sdfg
