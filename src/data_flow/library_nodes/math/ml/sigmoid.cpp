#include "sdfg/data_flow/library_nodes/math/ml/sigmoid.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

#include "sdfg/analysis/scope_analysis.h"

namespace sdfg {
namespace math {
namespace ml {

SigmoidNode::SigmoidNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<symbolic::Expression>& shape
)
    : ElementWiseUnaryNode(element_id, debug_info, vertex, parent, LibraryNodeType_Sigmoid, shape, {}) {}

bool SigmoidNode::expand_operation(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Sequence& body,
    const std::string& input_name,
    const std::string& output_name,
    const types::IType& input_type,
    const types::IType& output_type,
    const data_flow::Subset& subset
) {
    // Add code
    auto& code_block = builder.add_block(body);
    auto& input_node = builder.add_access(code_block, input_name);
    auto& output_node_neg = builder.add_access(code_block, output_name);
    auto& output_node_exp = builder.add_access(code_block, output_name);
    auto& output_node_add = builder.add_access(code_block, output_name);
    auto& output_node_div = builder.add_access(code_block, output_name);

    // -x
    {
        auto& tasklet = builder.add_tasklet(code_block, data_flow::TaskletCode::mul, "_out", {"-1.0f", "_in"});
        builder.add_computational_memlet(code_block, input_node, tasklet, "_in", subset, input_type);
        builder.add_computational_memlet(code_block, tasklet, "_out", output_node_neg, subset, output_type);
    }
    // exp(x)
    {
        auto& tasklet = builder.add_tasklet(code_block, data_flow::TaskletCode::expf, "_out", {"_in"});
        builder.add_computational_memlet(code_block, output_node_neg, tasklet, "_in", subset, output_type);
        builder.add_computational_memlet(code_block, tasklet, "_out", output_node_exp, subset, output_type);
    }

    // 1 + x
    {
        auto& tasklet = builder.add_tasklet(code_block, data_flow::TaskletCode::add, "_out", {"1.0f", "_in"});
        builder.add_computational_memlet(code_block, output_node_exp, tasklet, "_in", subset, output_type);
        builder.add_computational_memlet(code_block, tasklet, "_out", output_node_add, subset, output_type);
    }
    // 1.0f / x
    {
        auto& tasklet = builder.add_tasklet(code_block, data_flow::TaskletCode::div, "_out", {"1.0f", "_in"});
        builder.add_computational_memlet(code_block, output_node_add, tasklet, "_in", subset, output_type);
        builder.add_computational_memlet(code_block, tasklet, "_out", output_node_div, subset, output_type);
    }

    return true;
}

std::unique_ptr<data_flow::DataFlowNode> SigmoidNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<
        data_flow::DataFlowNode>(new SigmoidNode(element_id, this->debug_info(), vertex, parent, this->shape_));
}

} // namespace ml
} // namespace math
} // namespace sdfg
