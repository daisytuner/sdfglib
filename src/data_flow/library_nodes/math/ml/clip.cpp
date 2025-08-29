#include "sdfg/data_flow/library_nodes/math/ml/clip.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

#include "sdfg/analysis/scope_analysis.h"

namespace sdfg {
namespace math {
namespace ml {

ClipNode::ClipNode(
    size_t element_id,
    const DebugInfoRegion& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::string& min,
    const std::string& max
)
    : ElementWiseUnaryNode(element_id, debug_info, vertex, parent, LibraryNodeType_Clip, {{"min", min}, {"max", max}}) {
}

bool ClipNode::expand_operation(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Sequence& body,
    const std::string& input_name,
    const std::string& output_name,
    const types::IType& input_type,
    const types::IType& output_type,
    const data_flow::Subset& subset
) {
    std::string tmp_name = builder.find_new_name("__tmp");
    types::Scalar tmp_type(input_type.primitive_type());
    builder.add_container(tmp_name, tmp_type, false, false);

    // Add code
    auto& code_block = builder.add_block(body);
    auto& input_node_new = builder.add_access(code_block, input_name);
    auto& output_node_new = builder.add_access(code_block, output_name);
    auto& tmp_node = builder.add_access(code_block, tmp_name);

    // 1. Clip max
    if (!this->attributes_.at("max").empty()) {
        auto& tasklet =
            builder.add_tasklet(code_block, data_flow::TaskletCode::min, "_out", {"_in", this->attributes_.at("max")});
        builder.add_computational_memlet(code_block, input_node_new, tasklet, "_in", subset, input_type);
        builder.add_computational_memlet(code_block, tasklet, "_out", tmp_node, {}, output_type);
    } else {
        auto& tasklet = builder.add_tasklet(code_block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(code_block, input_node_new, tasklet, "_in", subset, input_type);
        builder.add_computational_memlet(code_block, tasklet, "_out", tmp_node, {}, tmp_type);
    }

    // 2. Clip min
    if (!this->attributes_.at("min").empty()) {
        auto& tasklet =
            builder.add_tasklet(code_block, data_flow::TaskletCode::max, "_out", {"_in", this->attributes_.at("min")});
        builder.add_computational_memlet(code_block, tmp_node, tasklet, "_in", {}, tmp_type);
        builder.add_computational_memlet(code_block, tasklet, "_out", output_node_new, subset, output_type);
    } else {
        auto& tasklet = builder.add_tasklet(code_block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(code_block, tmp_node, tasklet, "_in", {}, tmp_type);
        builder.add_computational_memlet(code_block, tasklet, "_out", output_node_new, subset, output_type);
    }

    return true;
}

std::unique_ptr<data_flow::DataFlowNode> ClipNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(new ClipNode(
        element_id, this->debug_info(), vertex, parent, this->attributes_.at("min"), this->attributes_.at("max")
    ));
}

} // namespace ml
} // namespace math
} // namespace sdfg
