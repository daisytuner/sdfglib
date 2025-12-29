#include "sdfg/data_flow/library_nodes/math/tensor/reduce_ops/max_node.h"

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"

namespace sdfg {
namespace math {
namespace tensor {

MaxNode::MaxNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<symbolic::Expression>& shape,
    const std::vector<int64_t>& axes,
    bool keepdims
)
    : ReduceNode(element_id, debug_info, vertex, parent, LibraryNodeType_Max, shape, axes, keepdims) {}

bool MaxNode::expand_reduction(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Sequence& body,
    const std::string& input_name,
    const std::string& output_name,
    const types::IType& input_type,
    const types::IType& output_type,
    const data_flow::Subset& input_subset,
    const data_flow::Subset& output_subset
) {
    auto& block = builder.add_block(body, {}, this->debug_info());

    auto& libnode = builder.add_library_node<math::cmath::CMathNode>(block, this->debug_info(), "fmax", 2);

    auto& in_access = builder.add_access(block, input_name, this->debug_info());
    auto& out_read_access = builder.add_access(block, output_name, this->debug_info());
    auto& out_write_access = builder.add_access(block, output_name, this->debug_info());

    builder.add_computational_memlet(block, in_access, libnode, "_in1", input_subset, input_type, this->debug_info());
    builder
        .add_computational_memlet(block, out_read_access, libnode, "_in2", output_subset, output_type, this->debug_info());
    builder.add_computational_memlet(
        block, libnode, "_out", out_write_access, output_subset, output_type, this->debug_info()
    );

    return true;
}

std::string MaxNode::identity() const { return "-INFINITY"; }

std::unique_ptr<data_flow::DataFlowNode> MaxNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(
        new MaxNode(element_id, this->debug_info(), vertex, parent, this->shape_, this->axes_, this->keepdims_)
    );
}

} // namespace tensor
} // namespace math
} // namespace sdfg
