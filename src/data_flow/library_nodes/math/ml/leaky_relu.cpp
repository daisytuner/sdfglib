#include "sdfg/data_flow/library_nodes/math/ml/leaky_relu.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

#include "sdfg/analysis/scope_analysis.h"

namespace sdfg {
namespace math {
namespace ml {

LeakyReLUNode::LeakyReLUNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<symbolic::Expression>& shape,
    const std::string& alpha
)
    : ElementWiseUnaryNode(element_id, debug_info, vertex, parent, LibraryNodeType_LeakyReLU, shape, {{"alpha", alpha}}) {
}

bool LeakyReLUNode::expand_operation(
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
    auto& output_node_max = builder.add_access(code_block, output_name);
    auto& output_node_mul = builder.add_access(code_block, output_name);

    // max(x, 0)
    {
        auto& tasklet = builder.add_tasklet(code_block, data_flow::TaskletCode::fma, "_out", {"0.0f", "_in"});
        builder.add_computational_memlet(code_block, input_node, tasklet, "_in", subset, input_type);
        builder.add_computational_memlet(code_block, tasklet, "_out", output_node_max, subset, output_type);
    }
    // alpha * x
    {
        auto& tasklet =
            builder.add_tasklet(code_block, data_flow::TaskletCode::mul, "_out", {this->attributes_.at("alpha"), "_in"});
        builder.add_computational_memlet(code_block, output_node_max, tasklet, "_in", subset, output_type);
        builder.add_computational_memlet(code_block, tasklet, "_out", output_node_mul, subset, output_type);
    }

    return true;
}

std::unique_ptr<data_flow::DataFlowNode> LeakyReLUNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(
        new LeakyReLUNode(element_id, this->debug_info(), vertex, parent, this->shape_, this->attributes_.at("alpha"))
    );
}

} // namespace ml
} // namespace math
} // namespace sdfg
